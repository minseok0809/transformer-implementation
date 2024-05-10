
import os
import torch
import random
import json
import tqdm
import logging
import datetime
import numpy as np
import tensorflow as tf
from pytz import timezone
import argparse
from prepare_data import load_data, json_to_txt, sentencepiece_tokenizer, tokenize_text
# from model import Transformer

parser = argparse.ArgumentParser(description='Transformers')
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)

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

    logger.info("*** Load Dataset ***")
    train_dataset, valid_dataset, test_dataset = load_data(args.data_dir)
    json_to_txt(args.data_dir)

    logger.info("*** Tokenize Dataset ***")
    src_lang = 'de'
    tgt_lang = 'en'
    vocab_size = 37000

    src_tokenizer_path = sentencepiece_tokenizer(args.data_dir, src_lang, vocab_size)
    tgt_tokenizer_path = sentencepiece_tokenizer(args.data_dir, tgt_lang, vocab_size)

    train_src_input_ids, train_tgt_input_ids = tokenize_text(args.data_dir, train_dataset, 
                                                             src_tokenizer_path, tgt_tokenizer_path, 
                                                             src_lang, tgt_lang)
    valid_src_input_ids, valid_tgt_input_ids = tokenize_text(args.data_dir, valid_dataset, 
                                                             src_tokenizer_path, tgt_tokenizer_path, 
                                                             src_lang, tgt_lang)
    test_src_input_ids, test_tgt_input_ids = tokenize_text(args.data_dir, test_dataset, 
                                                           src_tokenizer_path, tgt_tokenizer_path, 
                                                           src_lang, tgt_lang)



    for epoch in range(args.epoch):
        idx_list = list(range(0, train_src_input_ids.shape[0], args.batch_size))

        for (batch, idx) in enumerate(idx_list):
            epoch



@tf.function()
def train_step(src, tgt, model, optimizer):
    gold = tgt[:, 1:]
        
    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt)

    # 계산된 loss에 tf.GradientTape()를 적용해 학습을 진행합니다.
    with tf.GradientTape() as tape:
        predictions, enc_attns, dec_attns, dec_enc_attns = \
        model(src, tgt, enc_mask, dec_enc_mask, dec_mask)
        loss = loss_function(gold, predictions[:, :-1])

    # 최종적으로 optimizer.apply_gradients()가 사용됩니다. 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, enc_attns, dec_attns, dec_enc_attns
     

    # model = Transformer()     


if __name__ == '__main__':

    args = parser.parse_args()
    now_time = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H:%M'))
    args.time = now_time

    default_config = vars(args)

    with open(args.config, "w") as f:
        json.dump(default_config, f, indent=0)
    
    main(args)