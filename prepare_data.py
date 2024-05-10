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
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
        " --model_type=bpe" +
        " --vocab_size=37000" + 
        " --max_sentence_length=999999" + # 문장 최대 길이
        " --character_coverage=1.0"
        " --pad_id=0 --pad_piece=<pad>" + # pad (0)
        " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
        " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=</s>"  # end of sequence (3)
    )

    config = f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}\n" + \
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


def preprocess_function(examples : datasets, tokenizer, args) -> datasets:
    premise = examples['Title']
    hypothesis = examples['Content']
    # hypothesis2 = examples['Fake Content']
    label = examples['Label']
    
    # hypothesis2 = [" " if hyp2 == "not fake" else hyp2 for hyp2 in hypothesis2]

    if args.use_SIC:
        input_ids = tokenizer(premise, hypothesis, truncation=True, return_token_type_ids = False)['input_ids']
        length = [len(one_input) for one_input in input_ids]
        model_inputs = {'input_ids':input_ids, 'labels':label, 'length':length}
    else :
        # model_inputs = tokenizer(premise, hypothesis, hypothesis2, truncation=True, padding=True, return_token_type_ids = False)
        model_inputs = tokenizer(premise, hypothesis, max_length=512, truncation=True, padding=True, return_token_type_ids = False)
        model_inputs['labels'] = label

    return model_inputs


class TFDataset(Dataset):

    def __init__(self, bpm_model, tsv_file):
        sp = spm.SentencePieceProcessor()
        sp.load(bpm_model)

        self.sp = sp
        self.bos_id = sp.bos_id() #1
        self.eos_id = sp.eos_id() #2
        
        self.tsv_file = pd.read_csv(tsv_file, delimiter='\t', usecols=['src', 'tar'])
    
    def __len__(self):
        return len(self.tsv_file) #250k
    
    def __getitem__(self, idx):
        src_sent = self.tsv_file.iloc[idx, 0] 
        tar_sent = self.tsv_file.iloc[idx, 1]
        src_encoded = [self.bos_id] + self.sp.encode_as_ids(src_sent) + [self.eos_id]
        tar_encoded = [self.bos_id] + self.sp.encode_as_ids(tar_sent) + [self.eos_id]
        
        return torch.tensor(src_encoded), torch.tensor(tar_encoded)
    
def tokenize_text(data_dir, dataset, src_tokenizer_path, tgt_tokenizer_path, src_lang, tgt_lang):

    src_texts = []
    tgt_texts = []

    dataset = TFDataset(src_tokenizer_path, tgt_tokenizer_path, tsv_file)
    
    sampler = (DistributedSampler(dataset) if is_distributed else None)
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(is_distributed is False),
        sampler=sampler,
        collate_fn=collate_fn
    )
    return train_dataloader

    src_tokenizer = PreTrainedTokenizer(vocab_files_names=src_tokenizer_path)
    tgt_tokenizer = PreTrainedTokenizer(vocab_files_names=tgt_tokenizer_path)

    data = dataset['data']

    for idx in range(len(data)):
        src_texts.append(data[idx]['translation'][src_lang])
        tgt_texts.append(data[idx]['translation'][tgt_lang])


    # prepare_dataset
    # pbar.close()


    train_dataset = src_texts.map(
        prep_fn,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    tokenizer_folder = data_dir + "tokenizer/"
    with open(tokenizer_folder + "text_in_out_tokenizer.txt", "w") as f:
        f.write("\n")
        f.write("German Text"); f.write("\n")
        german_text = src_corpus[0]
        # german_text_not_pad = german_text[np.nonzero(german_text)]
        f.write('Input: {}'.format(german_text)); f.write("\n")
        f.write('Predicted Translation: {}'.format(tgt_tokenizer.DecodeIds(german_text)));  f.write("\n")
        
        f.write("\n")
        f.write("English Text"); f.write("\n")
        english_text = tgt_corpus[0]
        # english_text_not_pad = english_text[np.nonzero(english_text)]
        f.write('Input: {}'.format(english_text)); f.write("\n")
        f.write('Predicted Translation: {}'.format(tgt_tokenizer.DecodeIds(english_text))); f.write("\n")
    f.close()

    return src_input_ids, tgt_input_ids