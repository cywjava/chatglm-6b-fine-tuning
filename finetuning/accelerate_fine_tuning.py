# coding=UTF-8
import shutil
import time

from accelerate import Accelerator
import argparse
import os
import sys
from glob import glob

import torch
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModel, TrainingArguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune_util.alpaca_dataset import AlpacaDataset
from finetune_util.train_util import TrainUtil

"""
accelerate launch --gpu_ids='all' --config_file /home/train/.cache/huggingface/accelerate/default_config.yaml ./finetuning/accelerate_fine_tuning.py --model_path /home/train/model/ --dataset_path "/home/train/data/*" --check_points_path /home/train/check_points/ --train_batch_size 2 --epochs 2 --gradient_accumulation_steps 4 --fp16 
"""


def start_train(finetune_args):
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
                                                   sampler=train_sampler, drop_last=True,
                                                   collate_fn=train_util.data_collator)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=finetune_args.learning_rate)
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=finetune_args.epochs,
                              steps_per_epoch=len(train_data_loader))
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, eval_data_loader, lr_scheduler)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, train_data_loader,lr_scheduler)
    single_epoch_steps = len(train_data_loader)
    accelerator.print("*" * 100)
    accelerator.print(
        f"total epochs:{finetune_args.epochs},total steps:{int(finetune_args.epochs) * single_epoch_steps},single epoch steps:{single_epoch_steps}")
    accelerator.print("start train......")
    model.train()
    pt_name = "chatglm-6b-lora.pth"
    for epoch in tqdm(range(finetune_args.epochs), "Overall progress", colour="GREEN",
                      disable=not accelerator.is_local_main_process):
        with tqdm(range(single_epoch_steps), desc="Epoch " + str(epoch + 1) + " progress",
                  colour="GREEN", disable=not accelerator.is_local_main_process) as epoch_process_bar:
            for step, batch in enumerate(train_data_loader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    epoch_process_bar.update(round(step / single_epoch_steps * accelerator.num_processes, 2))

                    # if step % finetune_args.log_steps == 0 and step != 0:
                    #     accelerator.print(f"\nepoch:{(epoch + 1)},step:{step},loss:{loss}")
                    # if finetune_args.do_eval and step != 0:
                    #     accelerator.print("\neval loss:")
        save_pt(accelerator, model, finetune_args.check_points_path + os.sep + "epoch_" + str(epoch + 1), pt_name)
    save_pt(accelerator, model, finetune_args.check_points_path + os.sep + "final", pt_name)
    accelerator.print(f"\ntrain finished")


def save_pt(_accelerator, _model, pt_path, pt_name):
    _accelerator.wait_for_everyone()
    unwrapped_model = _accelerator.unwrap_model(_model)
    shutil.rmtree(pt_path, ignore_errors=True)
    os.makedirs(pt_path)
    time.sleep(1)
    _accelerator.save({
        k: v.to("cpu") for k, v in unwrapped_model.named_parameters() if v.requires_grad
    }, pt_path + os.sep + pt_name)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../data/*', type=str, required=False, help='数据集目录')
    parser.add_argument('--model_path', default="../model/", type=str, required=False, help='原始发布的预训练模型目录')
    parser.add_argument('--check_points_path', default="../check_points_path", type=str, required=False,
                        help='微调check_points_path保存目录')
    parser.add_argument('--epochs', default=50, type=int, required=False, help='训练epochs')
    parser.add_argument('--learning_rate', default=1e-4, type=float, required=False, help='learning_rate')
    parser.add_argument('--train_batch_size', default="4", type=int, required=False, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default="4", type=int, required=False, help='eval_batch_size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="梯度累积步数")
    parser.add_argument('--do_eval', action='store_true', help='do_eval')
    parser.add_argument('--log_steps', type=int, default=200)
    parser.add_argument('--debug', action='store_true', help='print dubug info')
    parser.add_argument('--fp16', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    start_train(set_args())
