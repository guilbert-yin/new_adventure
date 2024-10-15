formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time


deepspeed --include localhost:0 finetune.py \
    --model_name_or_path /root/autodl-fs/models/MiniCPM-2B-sft-bf16 \
    --output_dir output/mht/$formatted_time/ \
    --train_data_path data/mht_dataset_table_str_prompt_minicpm_refine1005_train_evi_num_top30.jsonl \
    --eval_data_path data/mht_dataset_table_str_prompt_minicpm_dev_evi_num_top30.jsonl \
    --learning_rate 1e-3 --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2  --model_max_length 2960 --bf16 --use_lora \
    --gradient_accumulation_steps 32 --warmup_steps 100 \
    --max_steps 1000 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 50 --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 1 --save_total_limit 3 | tee train.log