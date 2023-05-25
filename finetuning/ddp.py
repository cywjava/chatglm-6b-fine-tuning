# coding=UTF-8
import argparse
from glob import glob
import os

import sys

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune_util.alpaca_dataset import AlpacaDataset
from finetune_util.lora_trainer import LoraTrainer
from finetune_util.train_util import TrainUtil
import torch.distributed as dist
from torch.utils.data import DataLoader

"""
export CUDA_VISIBLE_DEVICES=0,2
export WORLD_SIZE=2
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=192.168.20.9

python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 ./finetuning/ddp.py --model_path /home/train/model/ --dataset_path "/home/train/data/*" --check_points_path /home/train/check_points/ --train_batch_size 3 --epochs 20 --fp16 --fp16_opt_level O2 --do_eval
"""


def start_train(finetune_args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend='nccl', init_method="tcp://localhost:29500", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    torch.cuda.empty_cache()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_util = TrainUtil(finetune_args, model, tokenizer)
    train_util.print_debug()

    # 生成训练集和测试集
    train_file_list = glob(pathname=finetune_args.dataset_path)
    # 2023-04-18 chenyiwan 重构loadset 操作
    train_dataset = AlpacaDataset(AlpacaDataset.load_json(train_file_list), tokenizer)
    eval_dataset = AlpacaDataset(train_dataset.eval_data(0.2), tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=finetune_args.train_batch_size,
                                                    sampler=train_sampler, drop_last=True)

    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_data_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=finetune_args.eval_batch_size,
                                                   sampler=eval_sampler, drop_last=True)

    args = TrainingArguments(
        output_dir=finetune_args.check_points_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=finetune_args.train_batch_size,
        per_device_eval_batch_size=finetune_args.eval_batch_size,
        do_eval=finetune_args.do_eval,
        evaluation_strategy="steps" if finetune_args.do_eval else "no",
        gradient_accumulation_steps=1,
        num_train_epochs=finetune_args.epochs,
        weight_decay=0.1,
        warmup_steps=1_000,
        learning_rate=finetune_args.learning_rate,
        fp16=finetune_args.fp16,
        fp16_opt_level=finetune_args.fp16_opt_level,
        push_to_hub=False,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        ignore_data_skip=True,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        auto_find_batch_size=True
    )

    len_dataset = len(train_dataset)
    batch_size = finetune_args.train_batch_size
    epochs = finetune_args.epochs
    # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epochs
    total_steps = (len_dataset // batch_size) * epochs if len_dataset % batch_size == 0 else (
                                                                                                     len_dataset // batch_size + 1) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_args.learning_rate)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    trainer = LoraTrainer(
        model=model,
        optimizers=(optimizer, lr_scheduler),
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_util.data_collator
    )
    print("start train...")
    trainer.train()
    # train_sampler.set_epoch(epoch)
    trainer.save_model(finetune_args.check_points_path + os.sep + "final_model")
    print("train finished...")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../data/*', type=str, required=False, help='数据集目录')
    parser.add_argument('--model_path', default="../model/", type=str, required=False, help='原始发布的预训练模型目录')
    parser.add_argument('--check_points_path', default="../check_points_path", type=str, required=False,
                        help='微调check_points_path保存目录')
    parser.add_argument('--epochs', default=50, type=int, required=False, help='训练epochs')
    parser.add_argument('--learning_rate', default=1e-3, type=float, required=False, help='learning_rate')
    parser.add_argument('--train_batch_size', default="4", type=int, required=False, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default="4", type=int, required=False, help='eval_batch_size')
    parser.add_argument('--do_eval', action='store_true', help='do_eval')
    parser.add_argument('--fp16', action='store_true', help='fp16')
    parser.add_argument('--fp16_opt_level', default="o2", type=str, required=False, help='fp16_opt_level')
    parser.add_argument('--debug', action='store_true', help='print dubug info')
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser.parse_args()


if __name__ == '__main__':
    start_train(set_args())
