accelerate launch --multi_gpu --num_machine=1 --num_processes=8 --main_process_port=8675 --mixed_precision="no" \
examples/summarization/run_summarization_no_trainer.py \
--model_name_or_path facebook/bart-large \
--dataset_name cnn_dailymail --dataset_config "3.0.0" \
--apply_lora --apply_rankselector \
--lora_type svd --target_rank 2 --lora_r 4 \
--lora_alpha 32 \
--reg_orth_coef 0.1 \
--init_warmup 5000 --final_warmup 85000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--per_device_train_batch_size 4 --learning_rate 5e-4   \
--num_train_epochs 15 --num_warmup_steps 3000 \
--max_source_length 1024 --max_target_length 160 --max_length 1024 \
--pad_to_max_length --num_beams 4 \
--per_device_eval_batch_size 4 \
--seed 9 \
--with_tracking \
--tb_writter_loginterval 500 \
--output_dir ./output/bart-large/cnn_dailymail  
