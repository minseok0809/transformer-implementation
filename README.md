# Transformer Implementation

<br><br>

## Introduction
This project is the unofficial implementation of NeurlIPS 2017 paper Attention is All You Need [[PDF](https://arxiv.org/abs/1706.03762)]. 
<br><br>


## Datasets
* IWSLT-2017-01: https://wit3.fbk.eu/2017-01
* IWSLT-2017-01-B: https://wit3.fbk.eu/2017-01-b
* iwslt2017: https://huggingface.co/datasets/IWSLT/iwslt2017
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

|  Model      | BLUE |
| :-----: |:----: |
| Transformer-Small  |  |  
| Transformer-Big |  | 
| T5-Small  | 19.21 | 
| T5-Big | 14.66 | 

<br>

|      Transformer-Small   | PPL | BLUE |
| :-----: |:----: | :----: | 
| Noam Optimizer + Label Smoothing | 60.11 | 16.87  | 
| Noam Optimizer | 22.97 | 16.99  |
| Adam Optimizer + Label Smoothing | 55.49  | 17.74  |
| Adam Optimizer | 21.75 | 17.69 |
| Adam Optimizer (Huggingface iwslt2017) |  |  |

<br>

|      Transformer-Big   | PPL | BLUE |
| :-----: |:----: | :----: |  
| Noam Optimizer + Label Smoothing |  |   |
| Noam Optimizer |  |   |
| Adam Optimizer + Label Smoothing |  |   |
| Adam Optimizer |  |   |

<br><br>
