# AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning

This pytorch package implements [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.10512.pdf) (ICLR 2023). 

**The implementaion of AdaLoRA has been merged to the parameter-efficient fine-tuning repository (🤗PEFT) supported by HuggingFace**: [🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft). Feel free to raise any issues when you using AdaLoRA in [PEFT](https://github.com/huggingface/peft) or our repository.   


## Repository Overview

There are several directories in this repo:

* [loralib/](loralib) contains the source code of the updated package `loralib`, which include our implementation of AdaLoRA ([loralib/adalora.py](loralib/loralib/adalora.py)) and needs to be installed to run the examples;
* [NLU/](NLU) contains an example implementation of AdaLoRA in DeBERTaV3-base, which produces the results on the GLUE benchmark;
* [NLG_QA/](NLG_QA) contains an example implementation of AdaLoRA in BART-large and DeBERTaV3-base, which can be used to reproduce the results of summarization and question-answering tasks. 


## Quickstart of AdaLoRA

1. Install the updated `loralib`:

  ```bash 
  pip install -e loralib/ 
  ```


2. Then we apply SVD-based adaptation of AdaLoRA. Here is an example (For more examples, please see [modeling_debertav2.py](NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py) for how we adapte DeBERTa): 

  ```python
  # ===== Before =====
  # layer = nn.Linear(in_features, out_features)
  
  # ===== After ======
  import loralib 
  # Add a SVD-based adaptation matrices with rank r=12
  layer = loralib.SVDLinear(in_features, out_features, r=12)
  ```

   Also, before the training loop begins, mark only LoRA parameters as trainable.
  ```python
  model = BigModel()
  # This sets requires_grad to False for all parameters without the string "lora_" in their names
  loralib.mark_only_lora_as_trainable(model)
  ```

3. During the training loop, we apply RankAllocator of AdaLoRA to update importance scores of incremental matrices and allocate budget accordingly. 
  ```python
  from loralib import RankAllocator
  from loralib import compute_orth_regu 
  # Initialize the RankAllocator 
  rankallocator = RankAllocator(
      model, lora_r=12, target_rank=8,
      init_warmup=500, final_warmup=1500, mask_interval=10, 
      total_step=3000, beta1=0.85, beta2=0.85, 
  )
  ```
+ `lora_r`: The initial rank of each incremental matrix. 
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `final_warmup`: The steps of final warmup for budget scheduler. 
+ `mask_interval`: The time internval between two budget allocations.
+ `beta1` and `beta2`: The coefficient of exponentional moving average when updating importance scores. 

  At each step of back-propagation, we apply an additional regularization to enforce the orthongonality of `SVDLinear` modules by `compute_orth_regu(model)`. After each step of `optimizer.step()`, we then call `RankAllocator` to update importance estimation and allocate the budget accordingly: 
  ```python
  # ===== Before =====
  # loss.backward() 
  # optimizer.step() 
  # global_step += 1 
  
  # ===== After ======
  (loss+compute_orth_regu(model, regu_weight=0.1)).backward
  optimizer.step()
  rankallocator.update_and_mask(model, global_step)
  global_step += 1
  ```


## GLUE benchmark

Check the folder `NLU` for more details about reproducing the GLUE results. 
An example of adapting DeBERTaV3-base on MNLI: 

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
NLU/examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--apply_adalora --apply_lora --lora_type svd \
--target_rank 1  --lora_r 3  \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 50000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 32 --learning_rate 5e-4 --num_train_epochs 7 \
--warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 30000 \
--logging_steps 500 \
--seed 6 \
--root_output_dir ./output/deberta-v3-base/mnli \
--overwrite_output_dir
```

Please see [`NLU/scripts`](NLU/scripts/) for more examples of GLUE. 


## Summarization and Question Answering Task

Check the folder [`NLG_QA`](NLG_QA/) for more details about reproducing the results of summarization and question-answering tasks.  
An example of adapting DeBERTaV3-base on SQuADv2: 

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
NLG_QA/examples/question-answering/run_qa.py \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad_v2 \
--apply_lora --apply_adalora \
--lora_type svd --target_rank 8   --lora_r 12  \
--reg_orth_coef 0.1 \
--init_warmup 50 --final_warmup 100 --mask_interval 10 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval --version_2_with_negative \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 8e-4 \
--num_train_epochs 1 \
--max_step 300 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 100000 \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 9 \
--root_output_dir ./output/debertav3-base/squadv2 \
--overwrite_output_dir 
```


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

