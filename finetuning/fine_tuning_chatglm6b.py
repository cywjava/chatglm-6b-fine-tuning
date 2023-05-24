# coding=UTF-8
import argparse
from glob import glob
import os

import sys

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel, TrainingArguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune_util.alpaca_dataset import AlpacaDataset
from finetune_util.lora_trainer import LoraTrainer
from finetune_util.multi_gpu import MultiGPU
from finetune_util.train_util import TrainUtil


def start_train(finetune_args):
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(finetune_args.model_path, trust_remote_code=True).half().cuda()
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
    MultiGPU().set_model_to_gpus(model)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    torch.cuda.empty_cache()

    train_util = TrainUtil(finetune_args, model, tokenizer)
    train_util.print_debug()

    # 生成训练集和测试集
    train_file_list = glob(pathname=finetune_args.dataset_path)
    # 2023-04-18 chenyiwan 重构loadset 操作
    train_dataset = AlpacaDataset(AlpacaDataset.load_json(train_file_list), tokenizer)
    eval_dataset = AlpacaDataset(AlpacaDataset.load_json(TrainUtil.build_validate_file(train_file_list, 0.2)),
                                 tokenizer)

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
        lr_scheduler_type="cosine",
        learning_rate=finetune_args.learning_rate,
        fp16=finetune_args.fp16,
        fp16_opt_level=finetune_args.fp16_opt_level,
        push_to_hub=False,
        remove_unused_columns=False,
        eval_steps=500,
        logging_steps=500,
        ignore_data_skip=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=True
    )

    trainer = LoraTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_util.data_collator
    )
    print("start train...")
    trainer.train()
    trainer.save_model(finetune_args.check_points_path + os.sep + "final_model")
    print("train finished...")


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
    parser.add_argument('--do_eval', action='store_true', help='do_eval')
    parser.add_argument('--fp16', action='store_true', help='fp16')
    parser.add_argument('--fp16_opt_level', default="O2", type=str, required=False, help='fp16_opt_level')
    parser.add_argument('--debug', action='store_true', help='print dubug info')
    return parser.parse_args()


if __name__ == '__main__':
    start_train(set_args())
