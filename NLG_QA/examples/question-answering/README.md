<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Question answering

This folder contains several scripts that showcase how to fine-tune a 🤗 Transformers model on a question answering dataset,
like SQuAD. 

## Setup
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r examples/summarization/requirements.txt 
pip install -e . 
pip install -e loralib/
export MASTER_PORT=8679
```

## Command for SQuADv2

Demo command for LoRA on DebertaV3-base: 

```
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 1   --lora_r 1   --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 2   --lora_r 2   --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 8   --lora_r 8   --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 16  --lora_r 16  --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 32  --lora_r 32  --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 58  --lora_r 58  --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 102 --lora_r 102 --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --lora_type frd --target_rank 156 --lora_r 156 --select_metric iptAB,magE,sumAB,onlyE --reg_orth_coef 0.0 --init_warmup 800000 --final_warmup 25000 --mask_interval 300 --finalize_rank --beta1 0.85 --beta2 0.00 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 8e-5 --num_train_epochs 8 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
```

Rankselection for DebertaV3-base: 
```
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 1   --lora_r 2   --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 2   --lora_r 4   --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 8   --lora_r 12  --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 16  --lora_r 25  --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 32  --lora_r 48  --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 58  --lora_r 68  --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 102 --lora_r 156 --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
python -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 examples/question-answering/run_qa.py --model_name_or_path microsoft/deberta-v3-base --dataset_name squad_v2 --apply_lora --apply_rankselector --lora_type svd --target_rank 156 --lora_r 198 --select_metric iptAB,iptE,sumAB,sumE --reg_orth_coef 0.1 --init_warmup 5000 --final_warmup 50000 --mask_interval 100 --finalize_rank --beta1 0.85 --beta2 0.85 --lora_module query,key,value,intermediate,layer.output,attention.output --lora_alpha 16 --do_train --do_eval --version_2_with_negative --max_seq_length 384 --doc_stride 128 --per_device_train_batch_size 16 --learning_rate 1e-3 --num_train_epochs 12 --warmup_steps 1000 --per_device_eval_batch_size 128 --evaluation_strategy steps --eval_steps 3000 --save_strategy steps --save_steps 100000 --logging_steps 300 --tb_writter_loginterval 300 --report_to tensorboard --seed 9 --root_output_dir /mnt/t-qingzhang/DataLog/debertav3-base/squadv2 --sub_output_dir curve --overwrite_output_dir 
```

## Previous Readme
## Trainer-based scripts

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py),
[`run_qa_beam_search.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search.py) and [`run_seq2seq_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py) leverage the 🤗 [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) for fine-tuning.

### Fine-tuning BERT on SQuAD1.0

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) script
allows to fine-tune any model from our [hub](https://huggingface.co/models) (as long as its architecture has a `ForQuestionAnswering` version in the library) on a question-answering dataset (such as SQuAD, or any other QA dataset available in the `datasets` library, or your own csv/jsonlines files) as long as they are structured the same way as SQuAD. You might need to tweak the data processing inside the script if your data is structured differently.

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks), if it doesn't you can still use the old version of the script which can be found [here](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering).

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag `--version_2_with_negative`.

This example code fine-tunes BERT on the SQuAD1.0 dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB.

```bash
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 88.52
exact_match = 81.22
```

### Fine-tuning XLNet with beam search on SQuAD

The [`run_qa_beam_search.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search.py) script is only meant to fine-tune XLNet, which is a special encoder-only Transformer model. The example code below fine-tunes XLNet on the SQuAD1.0 and SQuAD2.0 datasets.

#### Command for SQuAD1.0:

```bash
python run_qa_beam_search.py \
    --model_name_or_path xlnet-large-cased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_device_eval_batch_size=4  \
    --per_device_train_batch_size=4   \
    --save_steps 5000
```

#### Command for SQuAD2.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_qa_beam_search.py \
    --model_name_or_path xlnet-large-cased \
    --dataset_name squad_v2 \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_device_eval_batch_size=2  \
    --per_device_train_batch_size=2   \
    --save_steps 5000
```

### Fine-tuning T5 on SQuAD2.0

The [`run_seq2seq_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py) script is meant for encoder-decoder (also called seq2seq) Transformer models, such as T5 or BART. These
models are generative, rather than discriminative. This means that they learn to generate the correct answer, rather than predicting the start and end position of the tokens of the answer.

This example code fine-tunes T5 on the SQuAD2.0 dataset.

```bash
python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answer \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_seq2seq_squad/
```

## Accelerate-based scripts

Based on the scripts `run_qa_no_trainer.py` and `run_qa_beam_search_no_trainer.py`.

Like `run_qa.py` and `run_qa_beam_search.py`, these scripts allow you to fine-tune any of the models supported on a
SQuAD or a similar dataset, the main difference is that this script exposes the bare training loop, to allow you to quickly experiment and add any customization you would like. It offers less options than the script with `Trainer` (for instance you can easily change the options for the optimizer or the dataloaders directly in the script), but still run in a distributed setup, on TPU and supports mixed precision by leveraging the [🤗 `Accelerate`](https://github.com/huggingface/accelerate) library. 

You can use the script normally after installing it:

```bash
pip install git+https://github.com/huggingface/accelerate
```

then

```bash
python run_qa_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/debug_squad
```

You can then use your usual launchers to run in it in a distributed environment, but the easiest way is to run

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you can launch training with

```bash
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/debug_squad
```

This command is the same and will work for:

- a CPU-only setup
- a setup with one GPU
- a distributed training with several GPUs (single or multi node)
- a training on TPUs

Note that this library is in alpha release so your feedback is more than welcome if you encounter any problem using it.
