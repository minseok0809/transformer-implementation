import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import pad

import sentencepiece as spm
from pathlib import Path
import tensorflow as tf
import tqdm

import os
import argparse
import sentencepiece as spm

import glob
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_data(data_dir):
    json_paths = glob.glob(data_dir + "/*.json")
    for json_path in json_paths:
        if 'train' in json_path: train_data = {"data": json_path.split("/")[-1]}
        elif 'validation' in json_path: valid_data = {"data": json_path.split("/")[-1]}
        elif 'test' in json_path: test_data = {"data": json_path.split("/")[-1]}
   
    train_dataset = load_dataset(data_dir, data_files=train_data)
    valid_dataset = load_dataset(data_dir, data_files=valid_data)
    test_dataset = load_dataset(data_dir, data_files=test_data)

    return train_dataset, valid_dataset, test_dataset 


def json_to_txt(data_dir):

    train_dataset, valid_dataset, test_dataset = load_data(data_dir)

    train_de_texts = []; train_en_texts = []
    valid_de_texts = []; valid_en_texts = []
    test_de_texts = []; test_en_texts = []

    for text in train_dataset['data']['translation']:
        train_de_texts.append(text['de'])
        train_en_texts.append(text['en'])

    for text in valid_dataset['data']['translation']:
        valid_de_texts.append(text['de'])
        valid_en_texts.append(text['en'])

    for text in test_dataset['data']['translation']:
        test_de_texts.append(text['de'])
        test_en_texts.append(text['en'])

    txt_folder = data_dir + 'txt/'

    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    with open(os.path.join(txt_folder, 'train-de.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(train_de_texts)) 

    with open(os.path.join(txt_folder, 'train-en.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(train_en_texts)) 

    with open(os.path.join(txt_folder, 'valid-de.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(valid_de_texts)) 

    with open(os.path.join(txt_folder, 'valid-en.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(valid_en_texts)) 

    with open(os.path.join(txt_folder, 'test-de.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(test_de_texts)) 

    with open(os.path.join(txt_folder, 'test-en.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(test_en_texts)) 


def sentencepiece_tokenizer(data_dir, lang, vocab_size):

    tokenizer_folder = data_dir + "tokenizer/"
    txt_folder = data_dir + 'txt/'
    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)

    if lang == "de":
        path_end = "*de.txt"
        prefix = data_dir + "tokenizer/" + "transformer-sp-bpe-iwslt-de"
    elif lang == "en":
        path_end = "*en.txt"
        prefix = data_dir + "tokenizer/" + "transformer-sp-bpe-iwslt-en"
    
    paths = [str(x) for x in Path(txt_folder).glob(path_end)]
    corpus = ",".join(paths)

    vocab_size = vocab_size-7
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7} --minloglevel=2" + 
        " --model_type=bpe" +
        " --vocab_size=37000" + 
        " --max_sentence_length=999999" + # 문장 최대 길이
        " --character_coverage=1.0"
        " --pad_id=0 --pad_piece=<pad>" + # pad (0)
        " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
        " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=</s>"  # end of sequence (3)
    )

    config = f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7} --minloglevel=2\n " + \
        "--model_type=bpe\n" + \
        "--vocab_size=37000\n" + \
        "--max_sentence_length=999999\n" + \
        "--character_coverage=1.0" + \
        "--pad_id=0 --pad_piece=<pad>\n" + \
        "--unk_id=1 --unk_piece=<unk>\n" + \
        "--bos_id=2 --bos_piece=<s>\n" + \
        "--eos_id=3 --eos_piece=</s>\n"

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f'{prefix}.model')
    tokenizer_path = f'{prefix}.model'

    # print(config)
    # print(spm)

    return tokenizer_path


class TFDataset(Dataset):
       
    def __init__(self, src_tokenizer, tgt_tokenizer, src_texts, tgt_texts):

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_bos_id = src_tokenizer.bos_id()
        self.src_eos_id = src_tokenizer.eos_id()
        self.tgt_bos_id = tgt_tokenizer.bos_id() 
        self.tgt_eos_id = tgt_tokenizer.eos_id()
        
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self):
        return len(self.src_texts) 
    
    def __getitem__(self, idx):
        src_sent = self.src_texts[idx] 
        tgt_sent = self.tgt_texts[idx]
        src_encoded = [self.src_bos_id] + self.src_tokenizer.encode_as_ids(src_sent) + [self.src_eos_id]
        tgt_encoded = [self.tgt_bos_id] + self.tgt_tokenizer.encode_as_ids(tgt_sent) + [self.tgt_eos_id]
        
        src_tensor = torch.tensor(src_encoded)
        tgt_tensor = torch.tensor(tgt_encoded)


        return src_tensor, tgt_tensor
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def collate_fn(batch, max_pad=64):

    src_list, tgt_list, src_mask_list, tgt_mask_list  = [], [], [], []
    
    for (src, tgt) in batch:
        src_padded = pad(src, (0, max_pad - len(src))) # zero-padding to max_len 
        src_list.append(src_padded)
        tgt_padded = pad(tgt, (0, max_pad - len(tgt)))
        tgt_list.append(tgt_padded)

        src_seq_len = src_padded.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len))
        src_mask_list.append(src_mask)

        tgt_seq_len = tgt_padded.shape[0]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        tgt_mask_list.append(tgt_mask)

    src = torch.stack(src_list) # list([128],[128],[128]) => tensor w/ size([3,128])
    tgt = torch.stack(tgt_list)
    src_mask = torch.stack(src_mask_list) 
    tgt_mask = torch.stack(tgt_mask_list)

    return (src, tgt, src_mask, tgt_mask)


def text_tokenizer(src_tokenizer_path, tgt_tokenizer_path,
                   src_lang, tgt_lang, 
                   data_dir, dataset, batch_size, is_distributed=False):

    src_texts = []; tgt_texts = []

    src_tokenizer = spm.SentencePieceProcessor()
    src_tokenizer.load(src_tokenizer_path)

    tgt_tokenizer = spm.SentencePieceProcessor()
    tgt_tokenizer.load(tgt_tokenizer_path)

    data = dataset['data']
    
    for idx in range(len(data)):
        src_texts.append(data[idx]['translation'][src_lang])
        tgt_texts.append(data[idx]['translation'][tgt_lang])

    dataset = TFDataset(src_tokenizer, tgt_tokenizer, src_texts, tgt_texts)
    sampler = (DistributedSampler(dataset) if is_distributed else None)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(is_distributed is False),
        sampler=sampler,
        collate_fn=collate_fn
    )

    return dataloader


