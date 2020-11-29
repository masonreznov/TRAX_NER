"""
Author: Michael Salam
Date created: 2020/09/12
Last modified: 2020/09/12
Description: Contains the model description for the NER system using LSTM networks with TRAX library

"""

import trax
from trax import layers as tl
from trax.supervised import training
import os 
import numpy as np
import pandas as pd
from utils import get_params, get_vocab
import random as rnd



# the model
def NER(vocab_size=35181, d_model=50, tags=tag_map):
    '''
      Input: 
        vocab_size - integer containing the size of the vocabulary
        d_model - integer describing the embedding size
      Output:
        model - a trax serial model
    '''
    if args.vocab_size:
        vocab_size = args.vocab_size
    if args.d_model:
        d_model = args.d_model
    model = tl.Serial(
      tl.Embedding(vocab_size, d_model), # Embedding layer
      tl.LSTM(d_model), # LSTM layer
      tl.Dense(len(tags)), # Dense layer with len(tags) units
      tl.LogSoftmax()  # LogSoftmax layer
      )
    return model

