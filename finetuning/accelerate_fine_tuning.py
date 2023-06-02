# coding=UTF-8
import argparse
import os
import sys
from glob import glob

import torch
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune_util.alpaca_dataset import AlpacaDataset
from finetune_util.train_util import TrainUtil

"""
accelerate launch --gpu_ids='all' --config_file /home/train/.cache/huggingface/accelerate/default_config.yaml ./finetuning/accelerate_fine_tuning.py --model_path /home/train/model/ --dataset_path "/home/train/data/*" --check_points_path /home/train/check_points/ --train_batch_size 2 --epochs 2 --gradient_accumulation_steps 4 --fp16 
"""


def start_train(finetune_args):
    if finetune_args.debug:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"

    accelerator = Accelerator(gradient_accumulation_steps=finetune_args.gradient_accumulation_steps,
                              mixed_precision="fp16" if finetune_args.fp16 == True else "no")
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(finetune_args.model_path, trust_remote_code=True).cuda()
    else:
        model = AutoModel.from_pretrained(finetune_args.model_path, trust_remote_code=True).float()
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_key_value']
    )
    model = get_peft_model(model, peft_config)
    if accelerator.is_main_process:
        accelerator.print("*" * 100)
        model.print_trainable_parameters()
    model.enable_input_require_grads()
    torch.cuda.empty_cache()
    train_util = TrainUtil(finetune_args, model, tokenizer)
    # 生成训练集和测试集
    train_file_list = glob(pathname=finetune_args.dataset_path)
    # 2023-04-18 chenyiwan 重构loadset 操作
    train_dataset = AlpacaDataset(AlpacaDataset.load_json(train_file_list), tokenizer)
    eval_dataset = AlpacaDataset(train_dataset.eval_data(0.2), tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=42)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=finetune_args.train_batch_size,
                                                    shuffle=(train_sampler is None),
                                                    sampler=train_sampler, drop_last=True,
                                                    collate_fn=train_util.data_collator)

    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, seed=42)
    eval_data_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                   batch_size=finetune_args.eval_batch_size,
                                                   shuffle=(train_sampler is None),
                                                   sampler=eval_sampler, drop_last=True,
                                                   collate_fn=train_util.data_collator)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=finetune_args.learning_rate)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9999)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, eval_data_loader, lr_scheduler)

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0
    if finetune_args.resume_from_checkpoint:
        if finetune_args.resume_from_checkpoint is not None or finetune_args.resume_from_checkpoint != "":
            accelerator.print(f"\nResumed from checkpoint: {finetune_args.resume_from_checkpoint}")
            accelerator.load_state(finetune_args.resume_from_checkpoint)
            path = os.path.basename(finetune_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    accelerator.print("*" * 100)
    accelerator.print("start train......")
    pt_name = "chatglm-6b-lora.pt"
    for epoch in tqdm(range(starting_epoch, finetune_args.epochs), desc="Overall progress", colour="GREEN",
                      unit="epoch", disable=not accelerator.is_main_process):
        model.train()
        if finetune_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_data_loader, resume_step)
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_data_loader

        single_epoch_steps = len(active_dataloader)
        with tqdm(range(single_epoch_steps), desc="Epoch " + str(epoch + 1) + " progress", colour="GREEN", unit="step",
                  disable=not accelerator.is_main_process) as epoch_process_bar:
            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / finetune_args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if step % finetune_args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    overall_step += 1
                    epoch_process_bar.update(1)
                    if accelerator.is_main_process and finetune_args.checkpointing_steps != -1 and overall_step % finetune_args.checkpointing_steps == 0:
                        accelerator.print(f"\nstep:{overall_step},loss:{loss}\n")
                        save_pt(accelerator, model,os.path.join(finetune_args.check_points_path, f"step_{overall_step}"), pt_name)
            if accelerator.is_main_process:
                accelerator.print(f"\nstep:{overall_step},loss:{loss}\n")
                save_pt(accelerator, model, os.path.join(finetune_args.check_points_path, f"epoch_{(epoch + 1)}"),
                        pt_name)
    if accelerator.is_main_process:
        save_pt(accelerator, model, os.path.join(finetune_args.check_points_path, "final"), pt_name)
    accelerator.print(f"\ntrain finished")


def save_pt(_accelerator, _model, pt_path, pt_name):
    unwrapped_model = _accelerator.unwrap_model(_model)
    print(f"\nsaving checkpoint to directory:{pt_path}")
    if not os.path.exists(pt_path):
        os.mkdir(pt_path)
    torch.save({
        k: v.to("cpu") for k, v in unwrapped_model.named_parameters() if v.requires_grad
    }, pt_path + os.sep + pt_name)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../data/*', type=str, required=False, help='dateset directory')
    parser.add_argument('--model_path', default="../model/", type=str, required=False, help='mode directory')
    parser.add_argument('--check_points_path', default="../check_points_path", type=str, required=False,
                        help='checkpoint directory')
    parser.add_argument('--resume_from_checkpoint', default="", type=str, required=False,
                        help='Load previously saved model parameters from the specified checkpoint directory')
    parser.add_argument('--epochs', default=50, type=int, required=False, help='epochs')
    parser.add_argument('--learning_rate', default=1e-4, type=float, required=False, help='learning_rate')
    parser.add_argument('--train_batch_size', default="4", type=int, required=False, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default="4", type=int, required=False, help='eval_batch_size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="gradient_accumulation_steps")
    parser.add_argument('--do_eval', action='store_true', help='do_eval')
    parser.add_argument('--checkpointing_steps', type=int, default=500,
                        help='Whether the various states should be saved at the end of every n steps,if set to -1,no checkpoint is saved')
    parser.add_argument('--debug', action='store_true', help='Whether print nccl & torch distributed detail info')
    parser.add_argument('--fp16', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    start_train(set_args())
