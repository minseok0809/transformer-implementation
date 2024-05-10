import sentencepiece as spm
from pathlib import Path

from constants import *
from tqdm import tqdm

import os
import argparse
import sentencepiece as spm
from prepare_data import load_data


parser = argparse.ArgumentParser(description='Transformers')
parser.add_argument('--data_dir', type=str, default=None)

def json_to_txt(data_dir):

    train_dataset, valid_dataset, test_dataset = load_data(args.data_dir)

    train_german_texts = []; train_english_texts = []
    valid_german_texts = []; valid_english_texts = []
    test_german_texts = []; test_english_texts = []

    for text in train_dataset['data']['translation']:
        train_german_texts.append(text['de'])
        train_english_texts.append(text['en'])

    for text in valid_dataset['data']['translation']:
        valid_german_texts.append(text['de'])
        valid_english_texts.append(text['en'])

    for text in test_dataset['data']['translation']:
        test_german_texts.append(text['de'])
        test_english_texts.append(text['en'])

    folder_path = 'data/iwslt.17.de.en'
    with open(os.path.join(folder_path, 'train-de.txt'), "a", encoding='utf-8') as fp:        
        fp.write("\n".join(train_german_texts)) 

    with open(os.path.join(folder_path, 'train-en.txt'), "a", encoding='utf-8') as fp:        
        fp.write("\n".join(train_english_texts)) 

    with open(os.path.join(folder_path, 'valid-de.txt'), "a", encoding='utf-8') as fp:        
        fp.write("\n".join(valid_german_texts)) 

    with open(os.path.join(folder_path, 'valid-en.txt'), "a", encoding='utf-8') as fp:        
        fp.write("\n".join(valid_english_texts)) 

    with open(os.path.join(folder_path, 'test-de.txt'), "a", encoding='utf-8') as fp:        
        fp.write("\n".join(test_german_texts)) 

    with open(os.path.join(folder_path, 'test-en.txt'), "a", encoding='utf-8') as fp:        
        fp.write("\n".join(test_english_texts)) 


def sp(data_dir, is_src):

    paths = [str(x) for x in Path(data_dir).glob("*.txt")]
    corpus = ",".join(paths)
    prefix = "t5-sp-bpe-nsmc"
    vocab_size = 37000-7
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
        " --model_type=bpe" +
        " --vocab_size=37000" + 
        " --max_sentence_length=999999" + # 문장 최대 길이
        " --pad_id=0 --pad_piece=<pad>" + # pad (0)
        " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
        " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=</s>"  # end of sequence (3)
    )

    config = f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}\n" + \
        "--model_type=bpe\n" + \
        "--vocab_size=37000\n" + \
        "--max_sentence_length=999999\n" + \
        "--pad_id=0 --pad_piece=<pad>\n" + \
        "--unk_id=1 --unk_piece=<unk>\n" + \
        "--bos_id=2 --bos_piece=<s>\n" + \
        "--eos_id=3 --eos_piece=</s>\n"

    print(config)
    print(spm)

    if not os.path.isdir(SP_DIR):
        os.mkdir(SP_DIR)



if __name__=='__main__':


    args = parser.parse_args()
    default_config = vars(args)

    json_to_txt(args.data_dir)
 
    train_sp(is_src=True)
    train_sp(is_src=False)

    DATA_DIR = 'data/iwslt.17.de.en'
    SP_DIR = f'{DATA_DIR}/sp'
    SRC_DIR = 'src'
    TRG_DIR = 'trg'
    SRC_RAW_DATA_NAME = 'raw_data.src'
    TRG_RAW_DATA_NAME = 'raw_data.trg'
    TRAIN_NAME = 'train.txt'
    VALID_NAME = 'valid.txt'
    TEST_NAME = 'test.txt'

    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    src_model_prefix = 'src_sp'
    trg_model_prefix = 'trg_sp'
    sp_vocab_size = 16000
    character_coverage = 1.0
    model_type = 'unigram'


    split_data(SRC_RAW_DATA_NAME, SRC_DIR)
    split_data(TRG_RAW_DATA_NAME, TRG_DIR)
    