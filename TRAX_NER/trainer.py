"""
Author: Michael Salam
Date created: 2020/09/12
Last modified: 2020/09/12
Description: Contains the training function for the NER system using LSTM networks with TRAX library

"""
import trax
from trax import layers as tl
from trax.supervised import training
import os 
import numpy as np
import pandas as pd
from utils import get_params, get_vocab
from model import NER
from data_loader import data_generator
import argparse


# import the arguments
parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
parser.add_argument('--vocab-size', type=int, required=False,
                    help='the input size, if not given the extracted vocab size value is used')
parser.add_argument('--d-model', type=int, required=False,
                    help='LSTM embedding and hidden layer dimension')
parser.add_argument('--batch-size', type=int, required=True,
                    help='model batch size')
parser.add_argument('--train-steps', type=int, required=False,
                    help='total training steps, if not given the training will be done for a single step')
                    
parser.add_argument('--output-dir', metavar='path', type=str, required=True,
                    help='output path where model is written')
                    
args = parser.parse_args()



# make the output dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# data import
vocab, tag_map = get_vocab('data/large/words.txt', 'data/large/tags.txt')
t_sentences, t_labels, t_size = get_params(vocab, tag_map, 'data/large/train/sentences.txt', 'data/large/train/labels.txt')
v_sentences, v_labels, v_size = get_params(vocab, tag_map, 'data/large/val/sentences.txt', 'data/large/val/labels.txt')
test_sentences, test_labels, test_size = get_params(vocab, tag_map, 'data/large/test/sentences.txt', 'data/large/test/labels.txt')

# getting the vocab size, including the <PAD> token
vocab_size = len(vocab)

# initializing the model
model = NER(vocab_size=args.vocab_size, d_model=args.d_model, tags=tag_map)
batch_size = args.batch_size

# Create training data, mask pad id=35180 for training.
train_generator = trax.supervised.inputs.add_loss_weights(
    data_generator(batch_size, t_sentences, t_labels, vocab['<PAD>'], True),
    id_to_mask=vocab['<PAD>'])

# Create validation data, mask pad id=35180 for training.
eval_generator = trax.supervised.inputs.add_loss_weights(
    data_generator(batch_size, v_sentences, v_labels, vocab['<PAD>'], True),
    id_to_mask=vocab['<PAD>'])

def train_model(model, train_generator, eval_generator, train_steps=1, output_dir=args.output_dir):
    '''
    Input: 
        model - the model we are building
        train_generator - The data generator for training examples
        eval_generator - The data generator for validation examples,
        train_steps - number of training steps
        output_dir - folder to save your model
    Output:
        training_loop - a trax supervised training Loop
    '''
    train_task = training.TrainTask(
      train_generator, # A train data generator
      loss_layer = tl.CrossEntropyLoss(), # A cross-entropy loss function
      optimizer = trax.optimizers.Adam(0.01),  # The adam optimizer
    )

    eval_task = training.EvalTask(
      labeled_data = eval_generator, # A labeled data generator
      metrics = [tl.CrossEntropyLoss(), tl.Accuracy()], # Evaluate with cross-entropy loss and accuracy
      n_eval_batches = 10 # Number of batches to use on each evaluation
    )

    training_loop = training.Loop(
        model, # A model to train
        train_task, # A train task
        eval_task = eval_task, # The evaluation task
        output_dir = output_dir) # The output directory

    # Train with train_steps
    training_loop.run(n_steps = train_steps)
    return training_loop

# training loop
train_steps = args.train_steps            

# Train the model
training_loop = train_model(model, train_generator, eval_generator, train_steps)

