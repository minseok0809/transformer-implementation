import os
import json
import tqdm
import glob
import random
import evaluate
import datetime
import argparse
import logging
import pandas as pd 
import numpy as np
from pytz import timezone
import sentencepiece as spm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from thop import profile

from tensorboardX import SummaryWriter

from model.optimizer import NoamOpt
from model.transformer import TransformerForTranslation, TransformerOutput, Encoder, Decoder
from model.sublayer import MultiHeadAttention, ResidualConnection, FeedForwardNetwork
from model.embedding import WordEmbedding, InputEmbedding, PositionalEmbedding
from prepare_data import load_data, json_to_txt, SentencePieceTokenizer, TrainingDataset, InferenceDataset

summary = SummaryWriter()

parser = argparse.ArgumentParser(description='Transformers')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--test_dataset', default=None, type=str)
parser.add_argument('--inference_dataset', default=None, type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--config', default=None, type=str)
parser.add_argument('--load_model_path', default=None, type=str)
parser.add_argument('--data_dir', default=None, type=str)
parser.add_argument('--txt_dir', default=None, type=str)
parser.add_argument('--tokenizer_dir', default=None, type=str)
parser.add_argument('--tokenizer_model_name', default=None, type=str)
parser.add_argument('--encoding_type', default=None, type=str)
parser.add_argument('--src_lang', default=None, type=str)
parser.add_argument('--tgt_lang', default=None, type=str)
parser.add_argument('--max_seq_length', default=None, type=int)
parser.add_argument('--vocab_size', default=None, type=int)
parser.add_argument('--pad_id', default=None, type=int)
parser.add_argument('--unk_id', default=None, type=int)
parser.add_argument('--bos_id', default=None, type=int)
parser.add_argument('--eos_id', default=None, type=int)
parser.add_argument('--evaluation_metric', default=None, type=str)
parser.add_argument('--loss_type', default=None, type=str)
parser.add_argument('--seed', default=None, type=int) 
parser.add_argument('--epoch', default=None, type=int)
parser.add_argument('--logging_step', default=None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--sinusoidal_wave', default=None, type=int)
parser.add_argument('--embedding_dim', default=None, type=int)
parser.add_argument('--num_attention_heads', default=None, type=int)
parser.add_argument('--num_sub_layer', default=None, type=int)
parser.add_argument('--feed_forward_size', default=None, type=int)
parser.add_argument('--attention_dropout_prob', default=None, type=float)
parser.add_argument('--label_smoothing', default=None, type=float)
parser.add_argument('--optimizer_coefficient', default=None, type=float)
parser.add_argument('--warmup_steps', default=None, type=int)
parser.add_argument('--learning_rate', default=None, type=float)
parser.add_argument('--adam_beta1', default=None, type=float)
parser.add_argument('--adam_beta2', default=None, type=float)
parser.add_argument('--adam_epsilon', default=None, type=float)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('my-logger')
# logger.propagate = False

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

def main(args):

    seed_everything(args.seed)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # logger.info("*** Load Dataset ***")
    # train_dataset, valid_dataset, test_dataset = load_data(args.data_dir)
    # json_to_txt(args.data_dir, args.txt_dir)

    logger.info("*** Tokenize Dataset ***")

    src_tokenizer = SentencePieceTokenizer(args.tokenizer_dir,
                                           args.vocab_size // 2,
                                           args.encoding_type,
                                           args.pad_id,
                                           args.unk_id,
                                           args.bos_id,
                                           args.eos_id)
    tgt_tokenizer = SentencePieceTokenizer(args.tokenizer_dir,
                                           args.vocab_size // 2,
                                           args.encoding_type,
                                           args.pad_id,
                                           args.unk_id,
                                           args.bos_id,
                                           args.eos_id)

    src_paths = glob.glob(args.data_dir + "txt/" + "*" + args.src_lang + ".txt")
    src_paths = [src_path for src_path in src_paths if 'test' not in src_path ] 
    tgt_paths = glob.glob(args.data_dir + "txt/" + "*" + args.tgt_lang + ".txt")
    tgt_paths = [tgt_path for tgt_path in tgt_paths if 'test' not in tgt_path] 

    src_tokenizer.fit(",".join(src_paths), "/" + args.tokenizer_model_name + "-" + args.src_lang)
    tgt_tokenizer.fit(",".join(tgt_paths), "/" + args.tokenizer_model_name + "-" + args.tgt_lang)

    src_tokenizer.load_model("/" + args.tokenizer_model_name + "-" + args.src_lang + ".model")
    tgt_tokenizer.load_model("/" + args.tokenizer_model_name + "-" + args.tgt_lang + ".model")

    model = TransformerForTranslation(max_seq_length=args.max_seq_length,
                                      vocab_size=args.vocab_size // 2,
                                      embedding_dim=args.embedding_dim, 
                                      sinusoidal_wave=args.sinusoidal_wave,
                                      num_sub_layer=args.num_sub_layer,
                                      feed_forward_size=args.feed_forward_size,
                                      num_attention_heads=args.num_attention_heads,
                                      attention_dropout_prob=args.attention_dropout_prob).to(device)
    
    if args.loss_type == 'label_smoothing':
        criterion = nn.CrossEntropyLoss(ignore_index=args.pad_id, label_smoothing=args.label_smoothing)

    elif args.loss_type != 'label_smoothing':
        criterion = nn.NLLLoss(ignore_index=args.pad_id)

    if args.test_dataset == 'T':
        with open(args.data_dir + "/" + args.txt_dir + "test-" + args.src_lang + ".txt", 'r', encoding='utf-8') as f:
            src_lines = f.readlines()

        with open(args.data_dir + "/" + args.txt_dir + "test-" + args.tgt_lang + ".txt", 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()

    elif args.test_dataset == 'F':
        with open(args.inference_dataset + "_" + args.src_lang + ".txt", 'r', encoding='utf-8') as f:
            src_lines = f.readlines()

        with open(args.inference_dataset + "_" + args.tgt_lang + ".txt", 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()

    test_dataset = InferenceDataset(src_tokenizer=src_tokenizer,
                                    tgt_tokenizer=tgt_tokenizer,
                                    inputs=src_lines,
                                    outputs=tgt_lines,
                                    max_seq_length=args.max_seq_length,
                                    bos_id=args.bos_id)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)   
    

    metric = evaluate.load(args.evaluation_metric)

    model.load_state_dict(torch.load(args.load_model_path))
    model.eval()

    with torch.no_grad():
        sentence_bleu_scores = []
        df_predictions = []
        df_labels = []
        df_nested_labels = []

        test_loss = 0
        test_total = 0

        with tqdm.tqdm(test_dataloader) as pbar:
            for i, batch in enumerate(pbar):

                src_text, outputs, tgt_text = batch
                
                target = tgt_text
                bos_tokens = torch.ones(tgt_text.size()[0],1).long().to(device)*2 
                tgt_text = torch.cat((bos_tokens, tgt_text), dim=-1) 
                tgt_text = tgt_text[:,:-1]            

                if i == len(pbar) - 1:
                    print("\nPrediction:\n")

                for seq in range(args.max_seq_length):
                    output = model(src_text.to(device), outputs.to(device))
                    prediction = torch.argmax(output.to(device), dim=-1)[:,-1] 
                    outputs = torch.cat((outputs.to(device), prediction.to(device).view(-1,1)), dim=-1)
                    if i == len(pbar) - 1:
                        print(tgt_tokenizer.decode(outputs[0].tolist()))
                if i == len(pbar) - 1:
                    print()
                    print("Prediction ID:", tgt_tokenizer.decode(outputs[0].tolist()))
                    print()
                    print("Prediction Text:", outputs[0].tolist())
                    print("Label Text:", src_text[0].tolist())
                    print()

                perplexity_output = model(src_text.to(device), tgt_text.to(device))

                if args.loss_type == 'label_smoothing':
                    pass

                elif args.loss_type != 'label_smoothing':
                    softmax = nn.LogSoftmax(dim=-1)
                    perplexity_output = softmax(perplexity_output)
                    
                loss = criterion(perplexity_output.view(-1, len(tgt_tokenizer)), target.view(-1))

                test_loss += loss.item()
                test_total += 1     

                outputs = outputs.tolist()
                labels = tgt_text.tolist()

                clean_output = []
                clean_label = []
                for one_output, one_label in zip(outputs, labels):

                    try:
                        eos_idx = one_output.index(args.eos_id)
                        if eos_idx > 1:
                            one_output = one_output[1:eos_idx]
                        elif eos_idx == 1:
                            one_output = one_output[:eos_idx]
                            one_output[0] = 42
                    except:
                        pass
                    try:
                        eos_idx = one_label.index(args.eos_id)
                        if eos_idx > 1:
                            one_label = one_label[1:eos_idx]
                        elif eos_idx == 1:
                            one_label = one_label[:eos_idx]
                    except:
                        pass

                    clean_output.append(one_output)
                    clean_label.append(one_label)

                decoded_predictions = tgt_tokenizer.decode(clean_output)
                decoded_labels = tgt_tokenizer.decode(clean_label)

                for prediction, label in zip(decoded_predictions, decoded_labels):  


                    result = metric.compute(predictions=[prediction],
                                                            references=[[label]])
                    
                    if args.evaluation_metric == 'bleu':
                        sentence_bleu_score = result['bleu']
                    elif args.evaluation_metric == 'sacrebleu':
                        sentence_bleu_score = result['score'] 

                    df_labels.append(label)
                    df_predictions.append(prediction)   
                    df_nested_labels.append([label])  
                    sentence_bleu_scores.append(sentence_bleu_score)
        pbar.close()

    test_loss /= test_total
    test_perplexity = np.exp(test_loss)
    test_perplexity = round(test_loss, 4)       

    result = metric.compute(predictions=df_predictions,
                                            references=df_nested_labels)
    if args.evaluation_metric == 'bleu':
        corpus_bleu_score = result['bleu']
    elif args.evaluation_metric == 'sacrebleu':
        corpus_bleu_score = result['score'] 
        
    prediction_df = pd.DataFrame({'prediction':df_predictions,
                                    'label':df_labels,
                                    'bleu':sentence_bleu_scores})
    prediction_df.to_csv("{}{}.csv".format('./inference/',
                                                args.model_name),
                                                index=False)
    
    if args.evaluation_metric == 'bleu':
        corpus_bleu_score = corpus_bleu_score * 100
    elif args.evaluation_metric == 'sacrebleu':
        pass

    corpus_bleu_score = round(corpus_bleu_score, 4)
    print("Test BLEU Score:{}  Perplexity:{}".format(corpus_bleu_score, test_perplexity)); print("\n")

if __name__ == "__main__":

    args = parser.parse_args()
    now_time = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H:%M'))
    args.time = now_time

    default_config = vars(args)

    with open(args.config, "w") as f:
        json.dump(default_config, f, indent=0)
    
    main(args)

