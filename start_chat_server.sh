#!/bin/bash

cd /home/thudm/generate_server
ps -ef|grep "chat_server.py"|grep -v "grep"|awk '{print $2}'|xargs kill -9


export CUDA_VISIBLE_DEVICES="7"

nohup python3 chat_server.py --model_path /home/thudm/chatglm-6b --bot_type chat --sub_type normal --check_points_path /home/train/check_point/ --fp16  >> ./logs/chat_server.log 2>&1 &

