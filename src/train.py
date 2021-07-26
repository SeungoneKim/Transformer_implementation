import os
import sys
import argparse
import logging
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config.configs import set_random_fixed, get_path_info
from data.dataloader import get_dataloader
from data.tokenizer import Tokenizer
from util.utils import load_metricfn, load_optimizer, load_scheduler, load_lossfn, save_checkpoint, load_checkpoint, save_bestmodel, time_measurement, count_parameters
from models.model import build_model

class Trainer():
    def __init__(self, parser):
        
        # set parser
        self.args = parser.parse_args()

        # save loss history to plot later on
        self.training_history = []
        self.validation_history = []

        # set variables needed for training
        self.n_epoch = self.args.epoch
        self.train_batch_size = self.args.train_batch_size
        self.display_step = self.args.display_step # training
        self.val_batch_size = self.args.val_batch_size
        self.test_batch_size = self.args.test_batch_size
        self.display_examples = self.args.display_examples # testing
        
        self.lr = self.args.init_lr
        self.eps = self.args.adam_eps
        self.weight_decay = self.args.weight_decay
        self.beta1 = self.args.adam_beta1
        self.beta2 = self.args.adam_beta2

        self.warmup_steps = self.args.warm_up

        self.enc_language = self.args.enc_language
        self.dec_language = self.args.dec_language
        self.enc_max_len = self.args.enc_max_len
        self.dec_max_len = self.args.dec_max_len

        self.device = self.args.device

        # build dataloader
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloader(
            self.train_batch_size, self.val_batch_size, self.test_batch_size,
            self.enc_language, self.dec_language, self.enc_max_len, self.dec_max_len,
            self.args.dataset_name, self.args.dataset_type, self.args.category_name,
            self.args.x_name, self.args.y_name, self.args.percentage
        )
        self.train_batch_num = len(self.train_dataloader)
        self.val_batch_num = len(self.val_dataloader)
        self.test_batch_num = len(self.test_dataloader)
        
        self.t_total = self.train_batch_num * self.n_epoch

        # build tokenizer (for decoding purpose)
        self.decoder_tokenizer = Tokenizer(self.args.dec_language,self.args.dec_max_len)

        # load metric
        self.metric = load_metricfn(self.args.metric)
        
        # build model
        self.model = build_model(self.args.enc_pad_idx, self.args.dec_pad_idx,
                        self.args.enc_vocab_size, self.args.dec_vocab_size, 
                        self.args.model_dim, self.args.key_dim, self.args.value_dim, self.args.hidden_dim, 
                        self.args.num_heads, self.args.num_layers, self.args.enc_max_len, self.args.dec_max_len, self.args.drop_prob)

        # build optimizer
        self.optimizer = load_optimizer(self.model, self.lr, self.weight_decay, 
                                        self.beta1, self.beta2, self.eps)
        
        # build scheduler
        self.scheduler = load_scheduler(self.optimizer, self.warmup_steps, self.t_total)
        
        # build lossfn
        self.lossfn = load_lossfn(self.args.lossfn,self.args.dec_pad_idx)

    def train_test(self):
        best_model_epoch, training_history, validation_history = self.train()
        best_model = self.test(best_model_epoch)
        self.plot(training_history, validation_history)

    def train(self):
        
        # set logging
        logging.basicConfig(level=logging.INFO)
        
        # logging message
        logging.info('#################################################')
        logging.info('You have started training the model.')
        logging.info('Your model size is : ')
        logging.info(count_parameters(self.model))
        logging.info('#################################################')

        # set randomness of training procedure fixed
        self.set_random(516)
        
        # build directory to save to model's weights
        self.build_directory()

        # set initial variables for training, validation
        train_batch_num = len(self.train_dataloader)
        validation_batch_num = len(self.val_dataloader)

        # set initial variables for model selection
        best_model_epoch=0
        best_model_score=0
        best_model_loss =float('inf')

        # save information of the procedure of training
        training_history=[]
        validation_history=[]

        # predict when training will end based on average time
        total_time_spent_secs = 0
        
        # start of looping through training data
        for epoch_idx in range(self.n_epoch):
            # measure time when epoch start
            start_time = time.time()
            
            logging.info('#################################################')
            logging.info(f"Epoch : {epoch_idx+1} / {self.n_epoch}")
            logging.info('#################################################')

            ########################
            #### Training Phase ####
            ########################
            
            # switch model to train mode
            self.model.train()

            # set initial variables for training (inside epoch)
            training_loss_per_epoch=0.0
            training_score_per_epoch=0.0

            # train model using batch gradient descent with Adam Optimizer
            for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):
                # move batch of data to gpu
                encoder_input_ids = batch['encoder_input_ids'].to(self.device)
                encoder_attention_mask = batch['encoder_attention_mask'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                decoder_labels = batch['labels'].to(self.device)
                decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)

                # shift shape to (bs,sl)
                encoder_input_ids = encoder_input_ids.squeeze(1)
                decoder_input_ids = decoder_input_ids.squeeze(1)
                decoder_labels = decoder_labels.squeeze(1)

                # compute model output
                model_output = self.model(encoder_input_ids, decoder_input_ids[:, :-1]) # [bs,sl-1,vocab_dec]

                # reshape model output and labels
                reshaped_model_output = model_output.contiguous().view(-1,model_output.shape[-1]) # [bs*(sl-1),vocab_dec]
                reshaped_decoder_labels = decoder_labels[:,1:].contiguous().view(-1) # [bs*(sl-1)]
                
                # compute loss using model output and labels(reshaped ver)
                loss = self.lossfn(reshaped_model_output, reshaped_decoder_labels)

                # clear gradients, and compute gradient with current batch
                self.optimizer.zero_grad()
                loss.backward()

                # update gradients
                self.optimizer.step()

                # add loss to training_loss
                training_loss_per_iteration = loss.item()
                training_loss_per_epoch += training_loss_per_iteration

                # compute bleu score using model output and labels(reshaped ver)
                training_score_per_iteration = self.compute_bleu(reshaped_model_output,reshaped_decoder_labels)
                training_score_per_epoch += training_score_per_iteration["bleu"]

                # Display summaries of training procedure with period of display_step
                if ((batch_idx+1) % self.display_step==0) and (batch_idx>0):
                    print(f"Training Phase |  Epoch: {epoch_idx+1} |  Step: {batch_idx+1} / {train_batch_num} | loss : {training_loss_per_iteration} | score : {training_score_per_iteration['bleu']}")

            # update scheduler
            self.scheduler.step()

            # save training loss of each epoch, in other words, the average of every batch in the current epoch
            training_mean_loss_per_epoch = training_loss_per_epoch / train_batch_num
            training_history.append(training_mean_loss_per_epoch)

            ##########################
            #### Validation Phase ####
            ##########################

            # switch model to eval mode
            self.model.eval()

            # set initial variables for validation (inside epoch)
            validation_loss_per_epoch=0.0 
            validation_score_per_epoch=0.0

            # validate model using batch gradient descent with Adam Optimizer
            for batch_idx, batch in tqdm(enumerate(self.val_dataloader)):
                # move batch of data to gpu
                encoder_input_ids = batch['encoder_input_ids'].to(self.device)
                encoder_attention_mask = batch['encoder_attention_mask'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                decoder_labels = batch['labels'].to(self.device)
                decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)

                # shift shape to (bs,sl)
                encoder_input_ids = encoder_input_ids.squeeze(1)
                decoder_input_ids = decoder_input_ids.squeeze(1)
                decoder_labels = decoder_labels.squeeze(1)

                # compute model output
                model_output = self.model(encoder_input_ids, decoder_input_ids[:, :-1]) # [bs,sl-1,vocab_dec]
                
                # reshape model output and labels
                reshaped_model_output = model_output.contiguous().view(-1,model_output.shape[-1]) # [bs*(sl-1),vocab_dec]
                reshaped_decoder_labels = decoder_labels[:,1:].contiguous().view(-1) # [bs*(sl-1),vocab_dec]
                
                # compute loss using model output and labels(reshaped ver)
                loss = self.lossfn(reshaped_model_output, reshaped_decoder_labels)

                # add loss to training_loss
                validation_loss_per_iteration = loss.item()
                validation_loss_per_epoch += validation_loss_per_iteration

                # compute bleu score using model output and labels(reshaped ver)
                validation_score_per_iteration = self.compute_bleu(reshaped_model_output,reshaped_decoder_labels)
                validation_score_per_epoch += validation_score_per_iteration["bleu"]

            # save validation loss of each epoch, in other words, the average of every batch in the current epoch
            validation_mean_loss_per_epoch = validation_loss_per_epoch / validation_batch_num
            validation_history.append(validation_mean_loss_per_epoch)

            # save validation score of each epoch, in other words, the average of every batch in the current epoch
            validation_mean_score_per_epoch = validation_score_per_epoch / validation_batch_num

            # Display summaries of validation result after all validation is done
            logging.info(f"Validation Phase |  Epoch: {epoch_idx+1} | loss : {validation_mean_loss_per_epoch} | score : {validation_mean_score_per_epoch}")

            # Model Selection Process using validation_mean_score_per_epoch
            if (validation_mean_loss_per_epoch < best_model_loss):
                best_model_epoch = epoch_idx
                best_model_loss = validation_mean_loss_per_epoch
                best_model_score = validation_mean_score_per_epoch

                save_checkpoint(self.model, self.optimizer, epoch_idx,
                            os.path.join(self.args.weight_path,str(epoch_idx+1)+".pth"))

            # measure time when epoch end
            end_time = time.time()

            # measure the amount of time spent in this epoch
            epoch_mins, epoch_secs = time_measurement(start_time, end_time)
            logging.info(f"Time spent in {epoch_idx+1} is {epoch_mins} minuites and {epoch_secs} seconds")
            
            # measure the total amount of time spent until now
            total_time_spent += (end_time - start_time)
            total_time_spent_mins = int(total_time_spent/60)
            total_time_spent_secs = int(total_time_spent - (total_time_spent_mins*60))
            logging.info(f"Total amount of time spent until {epoch_idx+1} is {total_time_spent_mins} minuites and {total_time_spent_secs} seconds")

            # calculate how more time is estimated to be used for training
            avg_time_spent_secs = total_time_spent_secs / (epoch_idx+1)
            left_epochs = self.n_epoch - (epoch_idx+1)
            estimated_left_time = avg_time_spent_secs * left_epochs
            estimated_left_time_mins = int(estimated_left_time/60)
            estimated_left_time_secs = int(estimated_left_time - (estimated_left_time_mins*60))
            logging.info(f"Estimated amount of time until {self.n_epoch} is {estimated_left_time_mins} minuites and {estimated_left_time_secs} seconds")

        # summary of whole procedure    
        logging.info('#################################################')
        logging.info(f"Training and Validation has ended.")
        logging.info(f"Your best model was the model from epoch {best_model_epoch} and scored {self.args.metric} score : {best_model_score} and loss : {best_model_loss}")
        logging.info('#################################################')

        return best_model_epoch, training_history, validation_history
    
    def test(self, best_model_epoch):

        # logging message
        logging.info('#################################################')
        logging.info('You have started testing the model.')
        logging.info('#################################################')

        # set randomness of training procedure fixed
        self.set_random(516)

        # set weightpath
        weightpath = os.path.join(os.getcwd(),'weights')

        # loading the best_model from checkpoint
        best_model = build_model(self.args.pad_idx, self.args.pad_idx, self.args.bos_idx, 
                self.args.vocab_size, self.args.vocab_size, 
                self.args.model_dim, self.args.key_dim, self.args.value_dim, self.args.hidden_dim, 
                self.args.num_head, self.args.num_layers, self.args.max_len, self.args.drop_prob)
        
        load_checkpoint(best_model, self.optimizer, 
                    os.path.join(self.args.weight_path,str(best_model_epoch+1)+".pth"))

        # set initial variables for test
        test_batch_num = len(self.test_dataloader)

        ##########################
        ######  Test Phase  ######
        ##########################

        # switch model to eval mode
        best_model.eval()

        # set initial variables for testing
        test_score=0.0 
        
        # test model using batch gradient descent with Adam Optimizer
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.test_dataloader)):
                # move batch of data to gpu
                encoder_input_ids = batch['encoder_input_ids'].to(self.device)
                encoder_attention_mask = batch['encoder_attention_mask'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                decoder_labels = batch['labels'].to(self.device)
                decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)

                # shift shape to (bs,sl)
                encoder_input_ids = encoder_input_ids.squeeze(1)
                decoder_input_ids = decoder_input_ids.squeeze(1)
                decoder_labels = decoder_labels.squeeze(1)

                # compute model output
                best_model_output = best_model(encoder_input_ids, decoder_input_ids[:, :-1]) # [bs,sl-1,vocab_dec]
                
                # reshape model output and labels
                reshaped_best_model_output = best_model_output.contiguous().view(-1,best_model_output.shape[-1]) # [bs*(sl-1),vocab_dec]
                reshaped_decoder_labels = decoder_labels[:,1:].contiguous().view(-1) # [bs*(sl-1),vocab_dec]
                
                # compute bleu score using model output and labels(reshaped ver)
                test_score_per_iteration = self.compute_bleu(reshaped_best_model_output,reshaped_decoder_labels)
                test_score += test_score_per_iteration["bleu"]
                
                # Display examples of translation with period of display_examples
                if (batch_idx+1) % self.display_examples==0 and batch_idx>0:
                    # decode model_output and labels using Tokenizer
                    decoded_origins = self.decoder_tokenizer.decode(encoder_input_ids)
                    decoded_preds = self.decoder_tokenizer.decode(best_model_output)
                    decoded_labels = self.decoder_tokenizer.decode(decoder_labels)

                    # post process text for evaluation
                    decoded_origins = [origin.strip() for origin in decoded_origins]
                    decoded_preds = [pred.strip() for pred in decoded_preds]
                    decoded_labels = [label.strip() for label in decoded_labels]

                    # print out model_input(origin), model_output(pred) and labels(ground truth)
                    logging.info(f"Testing Phase | Step: {batch_idx+1} / {test_batch_num}")
                    logging.info("Original Sentence : ")
                    logging.info(decoded_origins)
                    logging.info("Ground Truth Translated Sentence : ")
                    logging.info(decoded_labels)
                    logging.info("Model Prediction - Translated Sentence : ")
                    logging.info(decoded_preds)

        # calculate test score
        test_score = test_score / test_batch_num

        # Evaluate summaries with period of display_steps
        logging.info(f"Test Phase |  Best Epoch: {best_model_epoch+1} | score : {test_score}")

        # save best model
        save_bestmodel(best_model,self.optimizer,self.args,
                            os.path.join(self.args.final_model_path,"bestmodel.pth"))

        # plot the result of training


        return best_model

    def plot(self, training_history, validation_history):
        step = np.linspace(0,self.n_epoch,self.n_epoch)
        plt.plot(step,np.array(training_history),label='Training')
        plt.plot(step,np.array(validation_history),label='Validation')
        plt.xlabel('number of epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def build_directory(self):
        # Making directory to store model pth
        curpath = os.getcwd()
        weightpath = os.path.join(curpath,'weights')
        os.mkdir(weightpath)

    def set_random(self, seed_num):
        set_random_fixed(seed_num)

    def compute_bleu(self, model_output, labels):
        # convert model_output into model_pred
        # [bs*(sl-1),vocab_dec] -> [bs*(sl-1)]
        model_pred = torch.argmax(model_output,dim=1)

        # decode model_output and labels using Tokenizer
        decoded_preds = self.decoder_tokenizer.decode(model_pred)
        decoded_labels = self.decoder_tokenizer.decode(labels)

        # post process text for evaluation
        # decoded_preds format : [ ["token1", "token2", ... , "tokenN"] ]
        # decoded_labels format : [ [["token1", "token2", ... , "tokenN"]] ]
        decoded_preds = [pred[0].strip() for pred in decoded_preds]
        decoded_labels = [label[0].strip() for label in decoded_labels]
        decoded_preds = [decoded_preds]
        decoded_labels = [[decoded_labels]]

        # compute bleu score
        result = self.metric.compute(predictions=decoded_preds,references=decoded_labels)
        result = {"bleu" : result['bleu']}

        # count the length of model_output
        prediction_lens = [np.count_nonzero(pred != self.decoder_tokenizer.pad_token) for pred in model_output]
        result['translation_length'] = np.mean(prediction_lens)

        # round the result with 4 digit precision
        result = {k: round(v,4) for k,v in result.items()}

        return result
