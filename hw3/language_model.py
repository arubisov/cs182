import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn

# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()
        
        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = your_code

        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        self.lstm = your_code
        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
#         self.dropout = your_code
        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output =       
        
    def forward(self,x):
        embeds = your_code
        lstm_out, _ = your_code
#         lstm_drop = your_code
#         logits = your_code
        logits = your_code
        return logits
