# Transformer Implementation

<br><br>

## Introduction
This project is the unofficial implementation of NeurlIPS 2017 paper Attention is All You Need [[PDF](https://arxiv.org/abs/1706.03762)]. 
<br><br>


## Data
* IWSLT-2017-01: https://wit3.fbk.eu/2017-01
* IWSLT-2017-01-B: https://wit3.fbk.eu/2017-01-b
* iwslt2017: https://huggingface.co/datasets/IWSLT/iwslt2017
<br><br>


## Model
* Transformer: https://arxiv.org/abs/1706.03762
* T5 Baseline: https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation
<br><br>

## Directory Structure
```
/
├── config/
├── data/
│   ├── iwslt17.de.en/
│   │   ├── tokenizer/
│   │   ├── txt/
│   │   ├── huggingface_txt/
│   │   ├── json/
│   └── iwslt17.de.en.orig/
│       └── xml/
├── model/
│   ├── __init__.py
│   ├── embedding.py
│   ├── optimizer.py
│   ├── sublayer.py
│   └── transformer.py
├── t5_baseline/
├── debug/
├── output/
├── inference/
│   ├── data
├── README.md
├── prepare_data.py
├── train.py
├── inference.py
├── data_preprocessing.ipynb
├── transformer_pipeline.ipynb
└── transformer_inference.ipynb
```

<br><br>


## Experiment result

<br>Data: Custom IWSLT 2017
<br>

|  Model      | PPL | BLUE |
| :-----: |:----: |:----: |
| Transformer-Small  | 21.75 | 17.69 | 
| Transformer-Big | 16.36 |  22.55 |
| T5-Small  | 8.20 | 19.21 |
| T5-Big | 164.77 | 14.66 |

<br>Data: HuggingFace iwslt2017
<br>

|  Model      | PPL | BLUE |
| :-----: |:----: |:----: |
| Transformer-Small | 9.13 | 26.35 |  
| Transformer-Big | 9.32 | 26.73 |
| T5-Small | 8.20 | 28.49 |
| T5-Big | 10.27 | 26.94 | 

<br>Model: Transformer-Small

|   Model      | PPL | BLUE |
| :-----: |:----: | :----: | 
| Noam Optimizer + Label Smoothing | 60.11 | 16.87  | 
| Noam Optimizer | 22.97 | 16.99  |
| Adam Optimizer + Label Smoothing | 55.49  | 17.74  |
| Adam Optimizer | 21.75 | 17.69 |
| Adam Optimizer (HuggingFace iwslt2017) | 9.13 | 26.35 |

<br>Model: Transformer-Big

|      Model   | PPL | BLUE |
| :-----: |:----: | :----: | 
| Adam Optimizer | 16.36 |  22.55 |
| Adam Optimizer (HuggingFace iwslt2017) | 9.32 | 26.73 |

<br><br>
