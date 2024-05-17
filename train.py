
import os
import glob
import json
import tqdm
import random
import logging
import datetime
import numpy as np

import torch
from torch import nn

from pytz import timezone
import argparse
import sentencepiece as spm

from model.optimizer import NoamOpt
from model.loss import LabelSmoothing
from model.transformer import TransformerForTranslation
from prepare_data import load_data, json_to_txt, sentencepiece_tokenizer, text_tokenizer

# from model import Transformer

parser = argparse.ArgumentParser(description='Transformers')
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--do_train', type=bool, default=False)
parser.add_argument('--do_eval', type=bool, default=False)
parser.add_argument('--do_predict', type=bool, default=False)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--debug_dir', type=str, default=None)
parser.add_argument('--src_lang', type=str, default=None)
parser.add_argument('--tgt_lang', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_seq_length', type=int, default=None)
parser.add_argument('--vocab_size', type=int, default=None)
parser.add_argument('--sinusoidal_wave', type=int, default=None)
parser.add_argument('--embedding_dim', type=int, default=None)
parser.add_argument('--num_sub_layer', type=int, default=None)
parser.add_argument('--feed_forward_size', type=int, default=None)
parser.add_argument('--num_attention_heads', type=int, default=None)
parser.add_argument('--attention_dropout_prob', type=float, default=None)

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

def get_blue_score():

    glob.glob(args.data_dir + )
    
    with open(args.data_dir + .format(type),encoding='utf-8') as f:
        inputs = f.readlines()
    with open('./dataset/de-en/test.en'.format(type),encoding='utf-8') as f:
        outputs = f.readlines()

    f.write('Input Text: {}'.format([tgt_tokenizer.DecodeIds(text.item()) for text in tgt_text[0]])); f.write("\n")
        
    bleu_score = 0.
    batch_size = 1
    for minibatch in [outputs[i:i + batch_size] for i in range(0, len(outputs), batch_size)]:
        predictions = translate(minibatch)
        for i in range(len(minibatch)):
            bleu_score += sentence_bleu([minibatch[i].strip().split()], predictions[i].strip().split())
    bleu_score /= len(outputs)
    return bleu_score


def main(args):

    seed_everything(args.seed)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    logger.info("*** Load Dataset ***")
    train_dataset, valid_dataset, test_dataset = load_data(args.data_dir)
    json_to_txt(args.data_dir)

    logger.info("*** Tokenize Dataset ***")
    src_tokenizer_path = sentencepiece_tokenizer(args.data_dir, args.src_lang, args.vocab_size)
    tgt_tokenizer_path = sentencepiece_tokenizer(args.data_dir, args.tgt_lang, args.vocab_size)

    src_tokenizer = spm.SentencePieceProcessor()
    src_tokenizer.load(src_tokenizer_path)

    tgt_tokenizer = spm.SentencePieceProcessor()
    tgt_tokenizer.load(tgt_tokenizer_path)

    train_dataloader = text_tokenizer(src_tokenizer_path, tgt_tokenizer_path, 
                                     args.src_lang, args.tgt_lang, 
                                     args.data_dir, train_dataset, args.batch_size)
    valid_dataloader = text_tokenizer(src_tokenizer_path, tgt_tokenizer_path, 
                                     args.src_lang, args.tgt_lang, 
                                     args.data_dir, valid_dataset, args.batch_size)
    test_dataloader = text_tokenizer(src_tokenizer_path, tgt_tokenizer_path, 
                                    args.src_lang, args.tgt_lang, 
                                    args.data_dir, test_dataset, args.batch_size)
    model = TransformerForTranslation(batch_size=args.batch_size,
                                      max_seq_length=args.max_seq_length,
                                      vocab_size=args.vocab_size,
                                      embedding_dim=args.embedding_dim, 
                                      sinusoidal_wave=args.sinusoidal_wave,
                                      num_sub_layer=args.num_sub_layer,
                                      feed_forward_size=args.feed_forward_size,
                                      num_attention_heads=args.num_attention_heads,
                                      attention_dropout_prob=args.attention_dropout_prob).to(device)
    
    criterion = LabelSmoothing(args.vocab_size, padding_idx=0, smoothing=0.1)
    optimizer = NoamOpt(args.embedding_dim, 0.1, 4000,
                        torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9), )
    
    if args.mode == "run":
        
        best_loss = float("inf")
        train_global_step = 0
        val_global_step = 0
        for epoch in range(args.epoch):
            train_loss = 0
            val_loss = 0
            train_total = 0
            val_total = 0

            if args.do_predict == True:
                model.train()
                with tqdm.tqdm(train_dataloader) as pbar:
                    pbar.set_description("Epoch " + str(epoch + 1))
                    for i, batch in enumerate(pbar):
                        src_text, tgt_text, src_mask, tgt_mask = batch
                        output = model(src_text, tgt_text, src_mask, tgt_mask)

                        loss = criterion(output.view(-1, len(tgt_tokenizer)), tgt_text.view(-1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_total += 1
                        train_global_step += 1
                train_loss /= train_total    
                pbar.close()
            
            if args.do_eval == True:
                model.eval()
                with torch.no_grad():
                    with tqdm.tqdm(valid_dataloader) as pbar:
                        pbar.set_description("Epoch " + str(epoch + 1))
                        for i, batch in enumerate(pbar):
                            src_text, tgt_text, src_mask, tgt_mask = batch
                            
                            output = model(src_text, tgt_text, src_mask, tgt_mask)

                            loss = criterion(output.view(-1, len(tgt_tokenizer)), tgt_text.view(-1))
                            val_loss += loss.item()*len(tgt_text)
                            val_total += 1
                            val_global_step +=1
                    val_loss /= val_total 
                    pbar.close()    
                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), "{}{}{}.pt".format(args.output_dir,
                                                                      args.model_name, 
                                                                      "_" + args.time))

                if args.do_predict == True:


    elif args.mode == "debug":
        for epoch in range(args.epoch):
            with tqdm.tqdm(train_dataloader) as pbar:
                pbar.set_description("Epoch " + str(epoch + 1))
                for i, batch in enumerate(pbar):
                    if i == 0:
                        first_batch = batch

                src_text, tgt_text, src_mask, tgt_mask = first_batch

                output = model(src_text, tgt_text, src_mask, tgt_mask)

                prediction = torch.argmax(output, dim=-1).tolist()[0]

            pbar.close()

        # with open(args.debug_dir + "text_tokenizer.txt", "w") as f:
        # with open(args.debug_dir + "input_embedding.txt", "w") as f:
        # with open(args.debug_dir + "position_embedding.txt", "w") as f:
        # with open(args.debug_dir + "multi_head_attention.txt", "w") as f:
        # with open(args.debug_dir + "single_head_attention.txt", "w") as f:
        # with open(args.debug_dir + "dot_product_attention.txt", "w") as f:
        # with open(args.debug_dir + "scaled_dot_product_attention.txt", "w") as f:
        # with open(args.debug_dir + "multi_head_attention_probablity.txt", "w") as f:
        # with open(args.debug_dir + "multi_head_attention_output.txt", "w") as f:
        # with open(args.debug_dir + "single_head_attention_output.txt", "w") as f:    
        # with open(args.debug_dir + "final_attention_output.txt", "w") as f: 
        # with open(args.debug_dir + "fead_foward_network_first_linear_layer.txt", "w") as f: 
        # with open(args.debug_dir + "fead_foward_network_activation_function.txt", "w") as f: 
        # with open(args.debug_dir + "fead_foward_network_second_linear_layer.txt", "w") as f: 
        # with open(args.debug_dir + "masked_scaled_dot_product_attention.txt", "w") as f: 
        # with open(args.debug_dir + "masked_attention_probablity.txt", "w") as f:  
        # with open(args.debug_dir + "masked_attention_output.txt", "w") as f:   
        # with open(args.debug_dir + "prediction_label_without_training.txt", "w") as f:              
        with open(args.debug_dir + "prediction_label_without_training.txt", "w") as f: 
        
            # f.write("output Size: {}".format(output.size())); f.write("\n\n")
            # f.write("output ID: {}".format(output)); f.write("\n")
            
            
            f.write("\n")
            f.write("German Text"); f.write("\n")
            f.write("Input ID: {}".format(src_text[0])); f.write("\n")
            f.write('Input Text: {}'.format([src_tokenizer.DecodeIds(text.item()) for text in src_text[0]])); f.write("\n")
            # f.write('Input Mask: {}'.format(src_mask[0])); f.write("\n")

            f.write("\n")
            f.write("English Text"); f.write("\n")
            f.write("Input ID: {}".format(tgt_text[0])); f.write("\n")
            f.write('Input Text: {}'.format([tgt_tokenizer.DecodeIds(text.item()) for text in tgt_text[0]])); f.write("\n")
            # f.write('Input Mask: {}'.format(tgt_mask[0])); f.write("\n")
            
            f.write("\n")
            f.write("Prediction Text"); f.write("\n")
            f.write("Output ID: {}".format(prediction)); f.write("\n")
            f.write('Output Text: {}'.format([tgt_tokenizer.DecodeIds(text) for text in prediction])); f.write("\n")
            
        f.close()


if __name__ == '__main__':

    args = parser.parse_args()
    now_time = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H:%M'))
    args.time = now_time

    default_config = vars(args)

    with open(args.config, "w") as f:
        json.dump(default_config, f, indent=0)
    
    main(args)