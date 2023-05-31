# coding=UTF-8
import argparse
import os
import sys
import time
from datetime import datetime
from glob import glob

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune_util.alpaca_dataset import AlpacaDataset
from finetune_util.lora_trainer import LoraTrainer
from finetune_util.train_util import TrainUtil
from apex import amp


def start_train(rank, world_size, finetune_args):
    torch.cuda.set_device(rank)
    if finetune_args.fp16:
        torch.set_default_dtype(torch.float16)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = AutoModel.from_pretrained(finetune_args.model_path, trust_remote_code=True).cuda()
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
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=[rank])
    train_util = TrainUtil(finetune_args, model, tokenizer)
    # 生成训练集和测试集
    train_file_list = glob(pathname=finetune_args.dataset_path)
    # 2023-04-18 chenyiwan 重构loadset 操作
    train_dataset = AlpacaDataset(AlpacaDataset.load_json(train_file_list), tokenizer)
    # eval_dataset = AlpacaDataset(train_dataset.eval_data(0.2), tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=42)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=finetune_args.train_batch_size,
                                                    shuffle=(train_sampler is None),
                                                    sampler=train_sampler, drop_last=True,
                                                    collate_fn=train_util.data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_args.learning_rate)
    print("start train...")
    for epoch in range(finetune_args.epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"epoch:{(epoch + 1)},start_time:{datetime.now()}")
        for step, batch in enumerate(train_data_loader):
            outputs = model(**batch)
            print(f"epoch:{(epoch + 1)},step:{(step + 1)}")

        if rank == 0:
            path = finetune_args.check_points_path + os.sep + (
                    "epoch_" + str(epoch + 1)) + os.sep + "chatglm-6b-lora.pt"
            print(f"save epoch:{(epoch + 1)} check point at:{path}")
            torch.save(model, path)
    if rank == 0:
        final_path = finetune_args.check_points_path + os.sep + "final" + os.sep + "chatglm-6b-lora.pt"
        print(f"save final check point at:{final_path}")
        torch.save(model, final_path)
        print(f"train finished")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../data/*', type=str, required=False, help='数据集目录')
    parser.add_argument('--model_path', default="../model/", type=str, required=False, help='原始发布的预训练模型目录')
    parser.add_argument('--check_points_path', default="../check_points_path", type=str, required=False,
                        help='微调check_points_path保存目录')
    parser.add_argument('--devices', default="0", type=str, required=False)
    parser.add_argument('--epochs', default=50, type=int, required=False, help='训练epochs')
    parser.add_argument('--learning_rate', default=1e-4, type=float, required=False, help='learning_rate')
    parser.add_argument('--train_batch_size', default="4", type=int, required=False, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default="4", type=int, required=False, help='eval_batch_size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="梯度累积步数")
    parser.add_argument('--do_eval', action='store_true', help='do_eval')
    parser.add_argument('--fp16', action='store_true', help='fp16')
    parser.add_argument('--fp16_opt_level', default="o2", type=str, required=False, help='fp16_opt_level')
    parser.add_argument('--debug', action='store_true', help='print dubug info')

    return parser.parse_args()


if __name__ == '__main__':
    _finetune_args = set_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["CUDA_VISIBLE_DEVICES"] = _finetune_args.devices
    _world_size = len(_finetune_args.devices.split(","))
    mp.spawn(start_train,
             args=(_world_size, _finetune_args,),
             nprocs=_world_size,
             join=True)
