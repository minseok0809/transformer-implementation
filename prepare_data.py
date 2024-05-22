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


def json_to_txt(data_dir, txt_dir):

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

    txt_dir = data_dir + txt_dir

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    with open(os.path.join(txt_dir, 'train-de.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(train_de_texts)) 

    with open(os.path.join(txt_dir, 'train-en.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(train_en_texts)) 

    with open(os.path.join(txt_dir, 'valid-de.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(valid_de_texts)) 

    with open(os.path.join(txt_dir, 'valid-en.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(valid_en_texts)) 

    with open(os.path.join(txt_dir, 'test-de.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(test_de_texts)) 

    with open(os.path.join(txt_dir, 'test-en.txt'), "w", encoding='utf-8') as fp:        
        fp.write("\n".join(test_en_texts)) 

class SentencePieceTokenizer(object):
    def __init__(self, tokenizer_path, vocab_size, encoding_type, pad_id, unk_id, bos_id, eos_id):
        
        self.templates = '--input={} --model_prefix={} --vocab_size={} --model_type={} --bos_id={} --eos_id={} --pad_id={} --unk_id={} --minloglevel=2'
        self.vocab_size = vocab_size
        self.encoding_type = encoding_type
        self.spm_path = tokenizer_path
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.l = 0
        self.alpha = 0
        self.n = 0    
            
    def transform(self, sentence, max_seq_length):
        if self.l and self.alpha:
            x = self.sp.SampleEncodeAsIds(sentence, self.l, self.alpha)
        elif self.n:
            x = self.sp.NBestEncodeAsIds(sentence, self.n)
        else:
            x = self.sp.EncodeAsIds(sentence)
        if max_seq_length > 0:
            pad = [0] * max_seq_length
            pad[:min(len(x), max_seq_length)] = x[:min(len(x), max_seq_length)]
            x = pad
        return x
    
    def fit(self, input_file, model_name):
        cmd = self.templates.format(input_file, self.spm_path + model_name, self.vocab_size, self.encoding_type,
                                    self.bos_id, self.eos_id, self.pad_id, self.unk_id)
        spm.SentencePieceTrainer.Train(cmd)
        
    def load_model(self, load_path):
        file = self.spm_path + load_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(file)
        self.sp.SetEncodeExtraOptions('bos:eos')
        return self

    def decode(self,encoded_sentences):
        decoded_output = []
        for encoded_sentence in encoded_sentences:
            x = self.sp.DecodeIds(encoded_sentence)
            decoded_output.append(x)
        return decoded_output
    
    def encode(self,decoded_sentences):
        encoded_output = []
        for decoded_sentence in decoded_sentences:
            x = self.sp.EncodeAsIds(decoded_sentence)
            encoded_output.append(x)
        return encoded_output

    def __len__(self):
        return len(self.sp)
    
class TrainingDataset(Dataset):
    def __init__(self, src_tokenizer, tgt_tokenizer, max_seq_length, data_dir, txt_dir, src_lang, tgt_lang, type='train'):
        
        tokenizer_dir = data_dir + "tokenizer/"
        txt_dir = data_dir + txt_dir
        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        src_paths = glob.glob(txt_dir + "*" + src_lang + ".txt")
        tgt_paths = glob.glob(txt_dir + "*" + tgt_lang + ".txt")

        for i in src_paths:
            if type in i:
                src_path = i

        for i in tgt_paths:
            if type in i:
                tgt_path = i    

        with open(src_path, encoding='utf-8') as f:
            src_line = f.readlines()
        with open(tgt_path, encoding='utf-8') as f:
            tgt_line = f.readlines()

        self.len = len(src_line)
        self.src = src_line
        self.tgt = tgt_line
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_length = max_seq_length
    
    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]
        src_tensor = torch.tensor(self.src_tokenizer.transform(src, 0), dtype=torch.float64, requires_grad=True).cuda()
        tgt_tensor = torch.tensor(self.tgt_tokenizer.transform(tgt, 0), dtype=torch.float64, requires_grad=True).cuda()

        return src_tensor, tgt_tensor
    
    def __len__(self):
        return self.len
    
    def collate_fn(self,data):
        def merge(sequences):
            padded_seqs = torch.zeros(len(sequences),self.max_seq_length, requires_grad=True).long().cuda()

            for i, seq in enumerate(sequences):
                padded_seqs[i][:min(self.max_seq_length,len(seq))] = seq[:min(self.max_seq_length,len(seq))]
            return padded_seqs

        data.sort(key=lambda x: len(x[0]), reverse=True)

        src_seqs, trg_seqs = zip(*data)
        src_seqs = merge(src_seqs)
        trg_seqs = merge(trg_seqs)

        return src_seqs, trg_seqs

class InferenceDataset(Dataset):
    def __init__(self, src_tokenizer, tgt_tokenizer, inputs, outputs,
                 max_seq_length, bos_id):

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.len = len(inputs)
        self.inputs = inputs
        self.outputs = outputs
        self.max_seq_length = max_seq_length
        self.bos_id = bos_id

    def __getitem__(self,index):
        input = self.inputs[index]
        label = self.outputs[index]
        inputs = torch.tensor(self.src_tokenizer.transform(input, self.max_seq_length)).cuda()
        outputs = torch.tensor([self.bos_id]) 
        labels = torch.tensor(self.tgt_tokenizer.transform(label, self.max_seq_length)).cuda()
        return inputs, outputs, labels

    def __len__(self):
        return self.len
    