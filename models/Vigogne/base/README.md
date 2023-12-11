---
license: llama2
base_model: TheBloke/vigogne-2-70B-chat-GPTQ
tags:
- generated_from_trainer
model-index:
- name: Vigogne70b-last_fan
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Vigogne70b-last_fan

This model is a fine-tuned version of [TheBloke/vigogne-2-70B-chat-GPTQ](https://huggingface.co/TheBloke/vigogne-2-70B-chat-GPTQ) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8007

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2
- training_steps: 2000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 0.04  | 200  | 0.9786          |
| No log        | 0.09  | 400  | 0.9424          |
| 0.9766        | 0.13  | 600  | 0.9057          |
| 0.9766        | 0.18  | 800  | 0.8812          |
| 0.8539        | 0.22  | 1000 | 0.8675          |
| 0.8539        | 0.27  | 1200 | 0.8434          |
| 0.8539        | 0.31  | 1400 | 0.8311          |
| 0.8396        | 0.36  | 1600 | 0.8195          |
| 0.8396        | 0.4   | 1800 | 0.8073          |
| 0.7841        | 0.44  | 2000 | 0.8007          |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.1.0+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0
