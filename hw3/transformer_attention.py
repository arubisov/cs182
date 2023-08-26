from typing import Optional, Callable, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
# from tensorflow.keras.layers import Layer

from transformer_utils import ApplyAttentionMask

class AttentionQKV(nn.Module):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self):
        super().__init__()
        self.apply_mask = ApplyAttentionMask()

    def forward(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        ####################################  YOUR CODE HERE  ####################################
        # n_queries corresponds to the sequence length on the query side
        # n_keyval corresponds to the sequence length on the key side (and value, as they are one and the same)
        # depth_k is the size of the projection that the key / query comparison is performed on.
        # depth_v is the size of the projection of the value projection. In a setting with one head, it is usually the dimension (dim) of the Transformer.
        # heads corresponds to the number of heads the attention is performed on.
        # If you are unfamiliar with attention heads, read section 3.2.2 of the Attention is all you need paper
         
        # PART 1: Implement Attention QKV
        # Use queries, keys and values to compute the output of the QKV attention

        # As defined is the Attention is all you need paper: https://arxiv.org/pdf/1706.03762.pdf
        key_dim = th.tensor(keys.shape[-1],dtype=th.float32)
        similarity =  # Compute the similarity according to the QKV formula

        masked_similarity = self.apply_mask(similarity, mask=mask) # We give you the mask to apply so that it is correct, you do not need to modify this.
        weights =  # Turn the similarity into a normalized output. Remember that the last dim contains the features
        output =  # Obtain the output
        ####################################  END OF YOUR CODE  ##################################

        return output, weights


class MultiHeadProjection(nn.Module):

    def __init__(self, n_heads, feature_sizes):
        """Map the multi-headed attention across the map

        Arguments:
            n_heads {int} -- The number of heads in the attention map
            feature_sizes {int} -- The size of the feature dimensions for key, query, and value

        """

        super().__init__()
        self.attention_map = AttentionQKV()
        self.n_heads = n_heads

        for size in feature_sizes:
            assert size % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def forward(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        queries, keys, values = inputs

        # Split each of the projection into its heads, by adding a new dimension
        # You must implement _split_heads, and _combine_heads
        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        # Apply the attention map
        attention_output_split, _ = self.attention_map(queries_split, keys_split, values_split, mask=mask)

        # Re-combine the heads together, and return the output.
        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, tensor):
        assert len(tensor.shape) == 3
        ####################################  YOUR CODE HERE  ####################################
        # PART 2: Implement the Multi-head attention.
        # You are given a Tensor which is one of the projections (K, Q or V)
        # and you must "split it" in self.n_heads. This splitting should add a dimension to the tensor,
        # so that each head acts independently

        batch_size, tensorlen = tensor.shape[0], tensor.shape[1]
        feature_size = tensor.shape[2]

        new_feature_size =  # Compute what the feature size per head is.
        # Reshape this projection tensor so that it has n_heads, each of new_feature_size
        tensor = 
        # Transpose the matrix so the outer-dimensions are the batch-size and the number of heads
        tensor = 
        return tensor
        ##########################################################################################

    def _combine_heads(self, tensor):
        assert len(tensor.shape) == 4
        ####################################  YOUR CODE HERE  ####################################
        # PART 2: Implement the Multi-head attention.
        # You are given the output from all the heads, and you must combine them back into 1 rank-3 matrix

        # Transpose back compared to the split, so that the outer dimensions are batch_size and sequence_length again
        tensor = 
        batch_size, tensorlen = tensor.shape[0], tensor.shape[1]
        feature_size = tensor.shape[-1]

        new_feature_size =  # What is the new feature size, if we combine all the heads
        tensor =  # Reshape the Tensor to remove the heads dimension and come back to a Rank-3 tensor
        return tensor
        ##########################################################################################

class MultiHeadAttention(nn.Module):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, n_heads, input_shapes):
        super().__init__()

        self.qa_channels, self.ma_channels = input_shapes

        self.n_heads = n_heads
        self.attention_layer = MultiHeadProjection(n_heads, (self.qa_channels,self.ma_channels))

        assert self.qa_channels % self.n_heads == 0 and self.ma_channels % self.n_heads == 0 and \
                                                        'Feature size must be divisible by n_heads'
        assert self.qa_channels == self.ma_channels and 'Cannot combine tensors with different shapes'

        self.query_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.key_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.value_layer = weight_norm(nn.Linear(self.ma_channels, self.ma_channels, bias=False))

        self.output_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))

        def weights_init(m):
            # if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
        self.query_layer.apply(weights_init)
        self.key_layer.apply(weights_init)
        self.value_layer.apply(weights_init)
        self.output_layer.apply(weights_init)


    def forward(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        assert (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 2 and \
                                                        'Must pass query and memory'
        query_antecedent, memory_antecedent = inputs
        q = self.query_layer(query_antecedent)
        k = self.key_layer(memory_antecedent)
        v = self.value_layer(memory_antecedent)

        attention_output = self.attention_layer((q, k, v), mask=mask)
        output = self.output_layer(attention_output)
        return output