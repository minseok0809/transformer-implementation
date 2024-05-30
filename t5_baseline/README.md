# T5 Baseline

<br><br>

## Introduction
Train & Evaluate T5 Scratch Model from Randomnized Weight Intialization

<br><br>


## Model
* t5-small-scratch-iwslt2017: https://huggingface.co/minseok0809/t5-small-scratch-iwslt2017
* t5-big-scratch-iwslt2017: https://huggingface.co/minseok0809/t5-big-scratch-iwslt2017
* t5-small-scratch-custom-iwslt2017: https://huggingface.co/minseok0809/t5-small-scratch-custom-iwslt2017
* t5-big-scratch-custom-iwslt2017: https://huggingface.co/minseok0809/t5-big-scratch-custom-iwslt2017

<br><br>

## Reference
* Transformer: https://arxiv.org/abs/1706.03762
* T5: https://arxiv.org/abs/1910.10683
* T5 Baseline: https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation

<br><br>

## Experiment result

<br>Data: Custom IWSLT 2017
<br>

|  Model      | PPL | BLUE |
| :-----: |:----: |:----: |
| T5-Small  | 8.20 | 19.21 |
| T5-Big | 164.77 | 14.66 |

<br>Data: HuggingFace iwslt2017
<br>

|  Model      | PPL | BLUE |
| :-----: |:----: |:----: |
| T5-Small | 8.20 | 28.49 |
| T5-Big | 10.27 | 26.94 | 

<br><br>

