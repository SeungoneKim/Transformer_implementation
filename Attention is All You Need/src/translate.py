import os
import sys
import argparse
import logging
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.tokenizer import Tokenizer
from util.utils import load_bestmodel

def translate(args, src_sentence, generated_max_length=100):
    # prepare best model
    best_model_path = os.path.join(args.final_model_path,"bestmodel.pth")
    try:
        best_model, _= load_bestmodel(best_model_path)
        best_model.eval()
    except OSError:
        logging.info("Check if there is bestmodel.pth file in final_results folder")

    # prepare device
    device = args.device

    # prepare tokenizer for both enc, dec
    enc_tokenizer = Tokenizer(args.enc_language,args.enc_max_len)
    dec_tokenizer = Tokenizer(args.dec_language,args.dec_max_len)

    # prepare vocabulary for both enc, dec
    enc_vocabulary = enc_tokenizer.get_vocab()
    dec_vocabulary = dec_tokenizer.get_vocab()

    # convert src_sentence, and measure the length of the src_sentence
    src_sentence = src_sentence.lower() # delete if you do not need cased
    src_sentence_length = len(src_sentence)

    logging.info(f"The original {args.enc_language} sentence you provided was : ")
    logging.info(src_sentence)
    
    # encode the given src_sentence with enc_tokenizer
    src_tensor = enc_tokenizer.encode(src_sentence).input_ids # [bs, sl]
    enc_mask = best_model.generate_padding_mask(src_tensor, src_tensor, "src", "src")

    logging.info(f"The {args.enc_language} Tokenizer converted sentence such as : ")
    logging.info(src_tensor)

    # prepare the pred_sentence
    pred_tensor=[dec_tokenizer.bos_token] # now : [1] -> goal : [generated_max_length]

    # translate the given sentence into target language
    with torch.no_grad():
        # pass through encoder
        encoder_output = best_model.Encoder(encoded_src_sentence, enc_mask) # [bs, sl, hs]

        for idx in range(generated_max_length):
            tgt_tensor = torch.LongTensor(pred_tensor).to(device)

            enc_dec_mask = best_model.geneate_padding_mask(tgt_tensor, enc_tensor, "src", "tgt")
            dec_mask = best_model.generate_padding_mask(tgt_tensor, tgt_tensor, "tgt", "tgt")

            # pass through decoder
            decoder_output = best_model.Decoder(tgt_tensor, encoder_output, enc_dec_mask, dec_mask) # [bs, sl, hs]

            # append predicted_token into pred_tensor
            predicted_token = output.argmax(dim=2)[:,-1].item()
            pred_tensor.append(predicted_token)

            # ENDING CONDITION : facing eos token
            if predicted_token == dec_vocabulary.eos_token :
                break
    
    # decode with dec_tokenizer
    translated_result = dec_tokenizer.decode(pred_tensor)
    translated_result = translated_result[0]

    # convert tensor into string
    translated_sentence = ""
    for tokens in translated_result:
        translated_sentence += tokens
        if tokens !='.':
            translated_sentence += " "

    return translated_sentence

