export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=4

JOB_NAME=YuLanMiniEval

python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=12345 \
    /home/chenzhipeng/yhy/train.py \
    --model_name_or_path /mnt/chenzhipeng/yhy/outputs/YuLanMiniTrain \
    --data_path /home/chenzhipeng/yhy/YuLan-Chat-Paddle-main/finetune/data/dev.jsonl \
    --bf16 True \
    --output_dir /mnt/chenzhipeng/yhy/outputs/${JOB_NAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed /home/chenzhipeng/yhy/configs/ds_z3_bf16.json \
&> /home/chenzhipeng/yhy/logs/${JOB_NAME}.log