# Adapating DeBERTaV3 with AdaLoRA for NLG and QA tasks

The folder contains the implementation of AdaLoRA in BART and DeBERTaV3 using the updated package of [`loralib`](../loralib/), which contains the implementation of AdaLoRA. AdaLoRA is present the following paper: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.10512.pdf) (ICLR 2023). 

## Setup Environment

### Create and activate the conda env
```bash
conda create -n NLG python=3.7
conda activate NLG 
```

### Install Pytorch
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install the pre-requisites
Install dependencies: 
```bash
pip install -r requirements.txt
```

Install `transformers`: (here we build our examples based on the latest `transformers` at the time we conduct experiments, which is `v4.21.0` and has better support for summarization tasks.)
```bash
pip install -e . 
```

Install the updated `loralib`:
```bash
pip install -e ../loralib/
```


## Adapt BART on summarization tasks

### The example to reproduce the XSum results

```bash
accelerate launch --multi_gpu --num_machine=1 --num_processes=8 \
--main_process_port=8679 --mixed_precision="no" \
examples/summarization/run_summarization_no_trainer.py \
--model_name_or_path facebook/bart-large \
--dataset_name xsum \
--apply_lora --apply_adalora \
--lora_type svd --target_rank 8 --lora_r 12 \
--lora_alpha 32 \
--reg_orth_coef 0.1 \
--init_warmup 6000 --final_warmup 25000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--per_device_train_batch_size 8 --learning_rate 5e-4 \
--num_train_epochs 25 --num_warmup_steps 3000 \
--max_source_length 768 --max_target_length 64 --max_length 768 \
--pad_to_max_length --num_beams 8 \
--per_device_eval_batch_size 8 \
--seed 9 \
--with_tracking \
--tb_writter_loginterval 500 \
--output_dir ./output/bart-large/xsum 
```


### Instructions

#### Hyperparameter Setup

+ `apply_lora`: Apply LoRA to the target model. 
+ `lora_type`: Config the low-rank parameterization, `frd` for low-rank decomposition and `svd` for SVD decomposition. Use `svd` for AdaLoRA and `frd` for LoRA. 
+ `apply_adalora`: Further apply AdaLoRA for the model that have been modified by LoRA. 
+ `lora_module`: The types of modules updated by LoRA. 
+ `lora_r`: The initial rank of each incremental matrix. 
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `final_warmup`: The steps of final warmup for budget scheduler. 
+ `mask_interval`: The time internval between two budget allocations.
+ `beta1` and `beta2`: The coefficient of exponentional moving average when updating importance scores. 
+ `reg_orth_coef`: The weight of orthongonal regularization. 


### Other examples

The floder `scripts` contains more examples of adapting BAET-large and DeBERTaV3-base with AdaLoRA on summarization and question-answering tasks.  


## Citation
```
@inproceedings{
    zhang2023adaptive,
    title={Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning },
    author={Qingru Zhang and Minshuo Chen and Alexander Bukharin and Pengcheng He and Yu Cheng and Weizhu Chen and Tuo Zhao},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=lq62uWRJjiY}
}
```