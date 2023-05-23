#!/bin/bash
export CUDA_VISIBLE_DEVICES="3"
cd /home/train/chatglm-6b-fine-tuning
rm -rf logs/*.log
nohup python3 -u ./finetuning/fine_tuning_chatglm6b.py --model_path /home/train/new_model/ --dataset_path "/home/train/data/*" --check_points_path /home/train/check_points/ --train_batch_size 3 --epochs 1 --do_eval --fp16 --fp16_opt_level O2 >> ./logs/train.log 2>&1 &