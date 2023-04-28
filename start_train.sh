#!/bin/bash
cd /home/train/chatglm-6b-fine-tuning

export CUDA_VISIBLE_DEVICES="0,1"
nohup python3 ./fine-tuning/fine_tuning_chatglm6b.py --model_path /home/train/model/ --dataset_path "/home/train/data/*" --check_points_path /home/train/check_points/ --train_batch_size 4 >>./logs/train.log 2>&1 &
