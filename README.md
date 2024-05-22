# Transformer Implementation

<br><br>

## Introduction
This project is the unofficial implementation of NeurlIPS 2017 paper Attention is All You Need [[PDF](https://arxiv.org/abs/1706.03762)]. 
<br><br>


## Datasets
* IWSLT-2017-01: https://wit3.fbk.eu/2017-01
* IWSLT-2017-01-B: https://wit3.fbk.eu/2017-01-b
<br><br>

## Directory Structure
```
/
├── config/
│   └── demo.json
├── data/
│   ├── iwslt17.de.en/
│   │   ├── tokenizer/
│   │   │   ├── transformer-sp-bpe-iwslt-de.model
│   │   │   ├── transformer-sp-bpe-iwslt-de.vocab
│   │   │   ├── transformer-sp-bpe-iwslt-en.model
│   │   │   └── transformer-sp-bpe-iwslt-en.vocab
│   │   ├── txt/
│   │   │   ├── test-de.txt
│   │   │   ├── test-en.txt
│   │   │   ├── train-de.txt
│   │   │   ├── train-en.txt
│   │   │   ├── valid-de.txt
│   │   │   └── valid-en.txt
│   │   ├── test-de-en.json
│   │   ├── train-de-en.json
│   │   └── validation-de-en.json
│   └── iwslt17.de.en.orig/
│       └── xml/
│           ├── dev/
│           │   ├── IWSLT17.TED.dev2010.de-en.de.xml
│           │   └── IWSLT17.TED.dev2010.de-en.en.xml
│           ├── test/
│           │   ├── IWSLT17.TED.tst2017.mltlng.de-en.de.xml
│           │   └── IWSLT17.TED.tst2017.mltlng.en-de.en.xml
│           └── train/
│               ├── train.tags.de-en.de.txt
│               ├── train.tags.de-en.de.xml
│               ├── train.tags.de-en.en.txt
│               └── train.tags.de-en.en.xml
├── model/
│   ├── __init__.py
│   ├── embedding.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── sublayer.py
│   └── transformer.py
├── debug/
├── output/
├── inference/
│   ├── data
├── README.md
├── prepare_data.py
├── train.py
├── inference.py
├── data_preprocessing.ipynb
└── transformer_pipeline.ipynb
```

<br><br>


## Experiment result

|         | BLUE |
| :-----: |:----: | 
| Transformer-Small  |  | 
| Transformer-Big |  |
| T5-Small  | 19.21 | 
| T5-Big | 14.66 |

<br><br>

|      Transformer-Small   | BLUE |
| :-----: |:----: | 
| Noam Optimizer + Label Smoothing |  | 
| Noam Optimizer |  |
| Adam Optimizer + Label Smoothing |  | 
| Adam Optimizer |  |

<br><br>
