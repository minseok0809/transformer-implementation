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
parser.add_argument('--train_eval_predict', default=None, type=str)
parser.add_argument('--loss_type', default=None, type=str)
parser.add_argument('--optimizer_type', default=None, type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--config', default=None, type=str)
parser.add_argument('--output_dir', default=None, type=str)
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

def int_to_exponential(num):
    return f"{num:.2e}"

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

    
    train_dataset = TrainingDataset(type="train",
                               src_tokenizer=src_tokenizer,
                               tgt_tokenizer=tgt_tokenizer,
                                max_seq_length=args.max_seq_length,
                                data_dir=args.data_dir,
                                txt_dir=args.txt_dir,
                                src_lang=args.src_lang,
                                tgt_lang=args.tgt_lang)
    
    valid_dataset = TrainingDataset(type="valid",
                            src_tokenizer=src_tokenizer,
                            tgt_tokenizer=tgt_tokenizer,
                            max_seq_length=args.max_seq_length,
                            data_dir=args.data_dir,
                            txt_dir=args.txt_dir,
                            src_lang=args.src_lang,
                            tgt_lang=args.tgt_lang)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=valid_dataset.collate_fn)
    
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

    if args.optimizer_type == 'noam':
        optimizer = NoamOpt(args.embedding_dim, args.optimizer_coefficient, args.warmup_steps,
                            torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon), )
        
    elif args.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                            lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2), 
                            eps=args.adam_epsilon)
                    
    with open(args.data_dir + "/" + args.txt_dir + "test-" + args.src_lang + ".txt", 'r', encoding='utf-8') as f:
        src_lines = f.readlines()

    with open(args.data_dir + "/" + args.txt_dir + "test-" + args.tgt_lang + ".txt", 'r', encoding='utf-8') as f:
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

    if args.mode == "run":

        best_loss = float("inf")
        cnt = 0
        train_global_step = 0
        valid_global_step = 0

        epoch_list = []
        train_loss_list = []
        valid_loss_list = []
        corpus_epoch_list = []
        corpus_bleu_score_list = []

        for epoch in range(args.epoch):
            train_loss = 0
            valid_loss = 0
            train_total = 0
            valid_total = 0

            if args.do_train == "T":
        
                model.train()
                with tqdm.tqdm(train_dataloader) as pbar:
                    pbar.set_description("Epoch " + str(epoch + 1))
                    for i, batch in enumerate(pbar):
                        src_text, tgt_text = batch

                        target = tgt_text

                        bos_tokens = torch.ones(tgt_text.size()[0],1).long().to(device)*2 # 2 means sos token
                        tgt_text = torch.cat((bos_tokens, tgt_text), dim=-1) # insert bos token in front
                        tgt_text = tgt_text[:,:-1]

                        output = model(src_text, tgt_text)
  
                        loss = criterion(output.view(-1, len(tgt_tokenizer)), target.view(-1))

                        if args.optimizer_type == 'noam':
                            optimizer.optimizer.zero_grad()
                        elif args.optimizer_type == 'adam':
                            optimizer.zero_grad()
                        
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_total += 1

                        if (train_global_step + 1) % args.logging_step == 0:
                            summary.add_scalar('loss/loss_train', loss.item(), train_global_step) # tensorboard 
                            prediction = torch.argmax(output, dim=-1).tolist()[0]
                            label = target.tolist()[0]
                            try:
                                label_eos_index = label.index(args.eos_id)
                                label = label[1:label_eos_index]
                                prediction_eos_index = prediction.index(args.eos_id)
                                prediction = prediction[1:prediction_eos_index]
                            except:
                                label = label[1:]
                                prediction = prediction[1:]
    
                            label_list = tgt_tokenizer.decode(label)
                            prediction_list = tgt_tokenizer.decode(prediction)

                            print("\n")
                            print("Label: {}".format(label_list))
                            print("Prediction: {}".format(prediction_list))

                            prediction = ' '.join(prediction_list)
                            label = ' '.join(label_list)
                            result = metric.compute(predictions=[prediction], references=[[label]])
                            bleu_score = result['bleu']
                            bleu_score =  round(bleu_score * 100, 4)
                            print("Blue Score:", bleu_score)                    

                        train_global_step += 1
                pbar.close()

                train_loss /= train_total
                prediction = torch.argmax(output, dim=-1).tolist()[0]
                label = target.tolist()[0]
                try:
                    eos_index = label.index(args.eos_id)
                    label = label[1:eos_index]
                    eos_index = prediction.index(args.eos_id)
                    prediction = prediction[1:eos_index]                      
                except:
                    pass

                label_list = tgt_tokenizer.decode(label)
                prediction_list = tgt_tokenizer.decode(prediction)

                print("\n")
                print("Label: {}".format(label_list))
                print("Prediction: {}".format(prediction_list))

                prediction = ' '.join(prediction_list)
                label = ' '.join(label_list)
                result = metric.compute(predictions=[prediction], references=[[label]])
                if args.evaluation_metric == 'bleu':
                    sentence_bleu_score = result['bleu']
                elif args.evaluation_metric == 'sacrebleu':
                    sentence_bleu_score = result['score']  
                sentence_bleu_score = round(sentence_bleu_score * 100, 4)
                print("Sentence Blue Score:", sentence_bleu_score, 4)             
            
                if args.do_eval == "T":
                    model.eval()
                    with torch.no_grad():
                        with tqdm.tqdm(valid_dataloader) as pbar:
                            pbar.set_description("Epoch " + str(epoch + 1))
                            for i, batch in enumerate(pbar):

                                src_text, tgt_text = batch

                                target = tgt_text
                                bos_tokens = torch.ones(tgt_text.size()[0],1).long().to(device)*2 # 2 means sos token
                                tgt_text = torch.cat((bos_tokens, tgt_text), dim=-1) # insert bos token in front
                                tgt_text = tgt_text[:,:-1]

                                output = model(src_text, tgt_text)
                                
                                loss = criterion(output.view(-1, len(tgt_tokenizer)), target.view(-1))

                                valid_loss += loss.item()
                                valid_total += 1
 
                                
                            valid_loss /= valid_total

                            prediction = torch.argmax(output, dim=-1).tolist()[0]
                            label = target.tolist()[0]
                            try:
                                label_eos_idx = label.index(args.eos_id)
                                label = label[1:label_eos_idx]
                                prediction_eos_idx = prediction.index(args.eos_id)
                                prediction = prediction[1:prediction_eos_idx]
                            except:
                                label = label[1:]
                                prediction = prediction[1:]
        
                            label_list = tgt_tokenizer.decode(label)
                            prediction_list = tgt_tokenizer.decode(prediction)

                            print("\n")
                            print("Label: {}".format(label_list))
                            print("Prediction: {}".format(prediction_list))

                            prediction = ' '.join(prediction_list)
                            label = ' '.join(label_list)
                            result = metric.compute(predictions=[prediction], references=[[label]])
                            if args.evaluation_metric == 'bleu':
                                sentence_bleu_score = result['bleu']
                            elif args.evaluation_metric == 'sacrebleu':
                                sentence_bleu_score = result['score']  
                            sentence_bleu_score = round(sentence_bleu_score * 100, 4)
                            print("Sentence Blue Score:", sentence_bleu_score) 

                        pbar.close()             
                    
                    print()
                    print("Epoch {}/{}, Train Loss: {:.3f}, Validation Loss: {:.3f}".format(epoch+1, args.epoch, train_loss, valid_loss))
                    print()

                    epoch_list.append(epoch)
                    train_loss_list.append(train_loss)
                    valid_loss_list.append(valid_loss)

                    prediction_df = pd.DataFrame({'Epoch':epoch_list,
                                                'Train Loss':train_loss_list,
                                                'Validation Loss':valid_loss_list})
                    prediction_df.to_csv("{}{}.csv".format(args.output_dir, args.model_name + "_prediction"), 
                                         index=False)
                                       
                    if best_loss > valid_loss:
                        best_loss = valid_loss
                        if epoch < 9:
                            epoch_num = "0" + str(epoch + 1)
                        elif epoch >= 9:
                            epoch_num = str(epoch + 1)
                        torch.save(model.state_dict(), args.output_dir + "{}.pt".format(args.model_name + "_" + str(epoch_num)))

                        if args.do_predict == "T":

                            model.eval()

                            with torch.no_grad():
                                sentence_bleu_scores = []
                                df_predictions = []
                                df_labels = []
                                df_nested_labels = []
                                with tqdm.tqdm(test_dataloader) as pbar:
                                    pbar.set_description("Epoch " + str(epoch + 1))
                                    for i, batch in enumerate(pbar):
                                        
                                        src_text, outputs, tgt_text = batch

                                        for seq in range(args.max_seq_length):
                                            prediction = model(src_text.to(device), outputs.to(device))
                                            prediction = torch.argmax(prediction.to(device), dim=-1)[:,-1] # get final token
                                            outputs = torch.cat((outputs.to(device), prediction.to(device).view(-1,1)), dim=-1)
                                            
                                        outputs = outputs.tolist()
                                        labels = tgt_text.tolist()

                                        clean_output = []
                                        clean_label = []
                                        for one_output, one_label in zip(outputs, labels):
            
                                            try:
                                                eos_idx = one_output.index(args.eos_id)
                                                one_output = one_output[1:eos_idx]
                                            except:
                                                one_output = one_output[1:]
                                                    # print("len(i)=",len(i))
                                                    # print("no eos token found")
                                            try:
                                                eos_idx = one_label.index(args.eos_id)
                                                one_label = one_label[1:eos_idx]
                                            except:
                                                one_label = one_label[1:]
                                                    # print("len(i)=",len(i))
                                                    # print("no eos token found")
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
                                            sentence_bleu_score = round(sentence_bleu_score * 100, 4)
                                            sentence_bleu_scores.append(sentence_bleu_score)
                                            
                                        valid_loss /= valid_total

                                        prediction = torch.argmax(output, dim=-1).tolist()[0]
                                        label = target.tolist()[0]
                                        try:
                                            label_eos_idx = label.index(args.eos_id)
                                            label = label[1:label_eos_idx]
                                            prediction_eos_idx = prediction.index(args.eos_id)
                                            prediction = prediction[1:prediction_eos_idx]
                                        except:
                                            label = label[1:]
                                            prediction = prediction[1:]
                    
                                        label_list = tgt_tokenizer.decode(label)
                                        prediction_list = tgt_tokenizer.decode(prediction)                           

                            result = metric.compute(predictions=df_predictions,
                                                                    references=df_nested_labels)
                            if args.evaluation_metric == 'bleu':
                                corpus_bleu_score = result['bleu']
                            elif args.evaluation_metric == 'sacrebleu':
                                corpus_bleu_score = result['score'] 

                            if epoch < 9:
                                epoch_num = "0" + str(epoch + 1)
                            elif epoch >= 9:
                                epoch_num = str(epoch + 1)

                            prediction_df = pd.DataFrame({'prediction':df_predictions,
                                                            'label':df_labels,
                                                            'bleu':sentence_bleu_scores})
                            prediction_df.to_csv("{}{}{}.csv".format(args.output_dir,
                                                                        args.model_name, 
                                                                        "_" + epoch_num),
                                                                        index=False)
                            corpus_bleu_score = round(corpus_bleu_score * 100, 4)
                            print("Test BLEU Score:{}".format(corpus_bleu_score)); print("\n")

                            corpus_epoch_list.append(epoch + 1)
                            corpus_bleu_score_list.append(corpus_bleu_score)

                            bleu_score_df = pd.DataFrame({'Epoch':corpus_epoch_list,
                                                        'BLEU Score':corpus_bleu_score_list})
                            bleu_score_df.to_csv("{}{}.csv".format(args.output_dir, args.model_name + "_bleu_score"), 
                                                index=False)  
                            
                            
    elif args.mode == "flops":
        with tqdm.tqdm(test_dataloader) as pbar:
            for i, batch in enumerate(pbar):
                src_text, tgt_text = batch

                target = tgt_text

                bos_tokens = torch.ones(tgt_text.size()[0],1).long().to(device)*2 # 2 means sos token
                tgt_text = torch.cat((bos_tokens, tgt_text), dim=-1) # insert bos token in front
                tgt_text = tgt_text[:,:-1]
                if i == 0:
                    macs, params = profile(model, inputs=(src_text, tgt_text))
                    
            pbar.close()
             
            flops = macs * 2
            flops_exponential_format = int_to_exponential(flops)
            gflops = round(flops/1e9, 7) 
    
            print("\n")
            print("Parameters:%s \n" %(params))
            print("FLOPs:%s \n" %(flops_exponential_format))
            print("GFLOPs:%s \n" %(gflops))


    elif args.mode == "debug":
        with tqdm.tqdm(test_dataloader) as pbar:
            for i, batch in enumerate(pbar):
                src_text, tgt_text = batch

                target = tgt_text

                bos_tokens = torch.ones(tgt_text.size()[0],1).long().to(device)*2 # 2 means sos token
                tgt_text = torch.cat((bos_tokens, tgt_text), dim=-1) # insert bos token in front
                tgt_text = tgt_text[:,:-1]
                if i == 0:
                    output = model(src_text, tgt_text)

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
        with open(args.debug_dir + "multi_head_attention_probablity.txt", "w") as f: 
                
            f.write("output Size: {}".format(output.size())); f.write("\n\n")
            f.write("output ID: {}".format(output)); f.write("\n")
            
            """
            src_id = src_text.tolist()[0]
            src_text = src_tokenizer.decode(src_id)

            tgt_id = target.tolist()[0]
            tgt_text = tgt_tokenizer.decode(tgt_id)

            f.write("\n")
            f.write("German Text"); f.write("\n")
            f.write("Input ID: {}".format(src_id)); f.write("\n")
            f.write('Input Text: {}'.format(src_text)); f.write("\n")

            f.write("\n")
            f.write("English Text"); f.write("\n")
            f.write("Input ID: {}".format(tgt_id)); f.write("\n")
            f.write('Input Text: {}'.format(tgt_text)); f.write("\n")
            
            prediction_id = torch.argmax(output, dim=-1).tolist()[0]
            try:

                eos_index = prediction_id.index(3)
                prediction_id = prediction_id[:eos_index]                      
            except:
                pass

            prediction_text = tgt_tokenizer.decode(prediction_id)

            f.write("\n")
            f.write("Prediction Text"); f.write("\n")
            f.write("Output ID: {}".format(prediction_id); f.write("\n")
            f.write('Output Text: {}'.format(prediction_text); f.write("\n")
            """

                             
if __name__ == "__main__":

    args = parser.parse_args()
    now_time = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H:%M'))
    args.time = now_time

    truth_table = args.train_eval_predict.split("_")
    args.do_train = truth_table[0]
    args.do_eval = truth_table[1]
    args.do_predict = truth_table[2]

    default_config = vars(args)

    with open(args.config, "w") as f:
        json.dump(default_config, f, indent=0)
    
    main(args)

