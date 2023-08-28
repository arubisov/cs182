import numpy as np
from segtok import tokenizer
import torch
from torch import nn

class LanguageModel(nn.Module):
    """Create a basic LSTM for language modeling"""
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=False):
        super().__init__()

        self.V = vocab_size
        self.H = rnn_size

        # Create an embedding layer of shape [vocab_size, rnn_size]
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = nn.Embedding(num_embeddings=self.V,
                                      embedding_dim=self.H)

        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        self.lstm = nn.LSTM(input_size=self.H,    # The number of expected features in the input x
                            hidden_size=self.H,   # The number of features in the hidden state h
                            num_layers=num_layers,  # Number of recurrent layers in stacked RNN
                            bias=True,              # whether layer uses bias weights b_ih and b_hh
                            batch_first=True,       # input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
                            dropout=dropout,        # dropout layer after each LSTM layer (except last layer) with p=dropout
                            bidirectional=False)    # if True, becomes a bidirectional LSTM

        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
        self.dropout = nn.Dropout(p=dropout)

        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output = nn.Linear(in_features=self.H,
                                out_features=self.V)


    def forward(self, x):
        """Run a forward pass through the Language Model and return raw scores (logits).

        Args:
        - x (torch.Tensor): minibatch of input strings, dim (B, T) where
                            B = minibatch size
                            T = sequence length

        Returns:
        - logits (torch.Tensor): raw class scores of dim (B, T, V)
                                 where V = vocab size
        """

        B, T = x.shape

        # project the each input in minibatch into an embedding of hidden_size.
        # returns size (B, T, H)
        embeds = self.embedding(x)

        # pass embedding through LSTM
        # returns size (B, T, H)
        lstm_out, _ = self.lstm(embeds)

        # pass last LSTM output through a dropout layer.
        # returns size (B, T, H)
        lstm_drop = self.dropout(lstm_out)

        # pass LSTM output through an affine layer mapping the hidden_size to
        # the vocab size.
        # (B, T, H) -> (B*T, H) @ (H, V) -> (B*T, V) -> (B, T, V)
        logits = self.output(lstm_drop.reshape(B*T, self.H)).reshape(B, T, self.V)

        return logits


    def loss(self, pred, target, mask):
        """Cross-entropy loss on the predictions."""

        # look at the pytorch docs for `CrossEntropyLoss` and `permute`
        loss = nn.CrossEntropyLoss(reduction='none')

        # CrossEntropyLoss requires input shape (N, C, d_1, ..., d_K) for K-dimensional loss.
        # we have 1D loss at each timestep T, shape (B, T, V), so need to get into shape (B, V, T).
        pred = torch.permute(pred, (0, 2, 1))

        # our target contains class indices, shape (B, T)
        # output is (B, T)
        loss_tensor = loss(pred, target)

        # apply mask, drop the PADs.
        # output is (B, T)
        loss_masked = (loss_tensor * mask)

        # get loss for the batch.
        # average the loss across non-masked tokens, average across the batch.
        batch_loss = (loss_masked.sum(dim=1) / (loss_masked != 0).sum(dim=1)).mean()

        return batch_loss


    def check_accuracy(self, pred, target, mask):
        """Check batch accuracy: % of non-masked tokens where predicted class (vocab item) is equal to the target class"""
        return (torch.eq(prediction.argmax(dim=2,keepdim=False), batch_target).float()*batch_target_mask).sum() / batch_target_mask.sum()
