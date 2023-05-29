ps -ef | grep -iE "ddp|multiprocessing.spawn" | grep -v "grep" | awk '{print $2}' | xargs kill -9
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE="2"
rm -rf logs/ddp.log
rm -rf /home/train/ddp_check_points/*
nohup python3 ./finetuning/ddp.py --model_path "/home/train/model/" --dataset_path "/home/train/data/*" --check_points_path "/home/train/ddp_check_points/" --train_batch_size 3 --fp16 --gradient_accumulation_steps 4 --local_rank 0 >>./logs/ddp.log 2>&1 &
tail -f logs/ddp.log