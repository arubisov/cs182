from typing import Optional, List
from collections import namedtuple

import torch as th
from torch import nn
from torch.nn import functional as F

import transformer_utils
from transformer_utils import EmbeddingTranspose, get_device
from transformer_attention import MultiHeadAttention

class PositionEmbedding(nn.Module):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, hidden_size):
        super(PositionEmbedding, self).__init__()

        assert hidden_size % 2 == 0 and 'Model vector size must be even for sinusoidal encoding'
        power = th.arange(0, hidden_size, step=2, dtype=th.float32)[:] / hidden_size
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def forward(self, inputs, start=1):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        ####################################  YOUR CODE HERE  ####################################
        # PART 3: Implement the Position Embedding.
        # As stated in section 3.5 of the paper, attention does not naturally embed position information
        # To incorporate that, the authors use a variable frequency sin embedding.
        # Note that we use zero-indexing here while the authors use one-indexing

        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        batch_size = inputs.shape[0]
        sequence_length = inputs.shape[1]

        # obtain a sequence that starts at `start` and increments for `sequence_length `
        seq_pos = th.arange(start, sequence_length + start, dtype=th.float32)
        seq_pos_expanded = seq_pos[None,:,None]
        index = seq_pos_expanded.repeat(*[1,1,self.hidden_size//2])

        # create the position embedding as described in the paper
        # use the `divisor` attribute instantiated in __init__ 
        sin_embedding = 
        cos_embedding = 

        # interleave the sin and cos. For more info see:
        # https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/3
        position_shape = (1, , ) # fill in the other two dimensions
        position_embedding = th.stack((sin_embedding,cos_embedding), dim=3).view(position_shape)

        pos_embed_deviced = position_embedding.to(get_device())
        return  # add the embedding to the input
        ####################################  END OF YOUR CODE  ##################################

class TransformerFeedForward(nn.Module):
    def __init__(self, input_size,
                 filter_size,
                 hidden_size,
                 dropout):
        super(TransformerFeedForward, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.feed_forward = nn.Sequential(
                                nn.Linear(input_size,filter_size),
                                nn.ReLU(),
                                nn.Linear(filter_size,hidden_size)
                            )
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.feed_forward.apply(weights_init)
        self.dropout = nn.Dropout(0 if dropout is None else dropout)

    def forward(self, inputs):
        ####################################  YOUR CODE HERE  ####################################
        # PART 4.1: Implement the FeedForward Layer.
        # As seen in fig1, the feedforward layer includes a normalization and residual
        norm_input = 
        dense_out = 
        dense_drop =  # Add the dropout here
        return  # Add the residual here
        ####################################  END OF YOUR CODE  ##################################


class TransformerEncoderBlock(nn.Module):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads,[input_size,input_size])
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, inputs, self_attention_mask=None):

        ####################################  YOUR CODE HERE  ####################################
        # PART 4.2: Implement the Transformer Encoder according to section 3.1 of the paper.
        # Perform a multi-headed self-attention across the inputs.

        # First normalize the input with the LayerNorm initialized in the __init__ function (self.norm)
        norm_inputs = 

        # Apply the self-attention with the normalized input, use the self_attention mask as the optional mask parameter.
        attn = 

        # Apply the residual connection. res_attn should sum the attention output and the original, non-normalized inputs
        res_attn =  # Residual connection of the attention block

        # output passes through a feed_forward network
        output = 
        return output


class TransformerDecoderBlock(nn.Module):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def __init__(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads,[input_size,input_size])

        self.cross_attention = MultiHeadAttention(n_heads,[input_size,input_size])
        self.cross_norm_source = nn.LayerNorm(input_size)
        self.cross_norm_target = nn.LayerNorm(input_size)
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, decoder_inputs, encoder_outputs, self_attention_mask=None, cross_attention_mask=None):    
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        ####################################  YOUR CODE HERE  ####################################
        # PART 4.2: Implement the Transformer Decoder according to section 3.1 of the paper.
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        # Compute the selt-attention over the decoder inputs. This uses the self-attention
        # mask to control for the future outputs.
        # This generates a tensor of size [batch_size x target_len x d_model]

        norm_decoder_inputs = 

        target_selfattn = 
        res_target_self_attn = 

        # Compute the attention using the keys/values from the encoder, and the query from the
        # decoder. This takes the encoder output of size [batch_size x source_len x d_model] and the
        # target self-attention layer of size [batch_size x target_len x d_model] and then computes
        # a multi-headed attention across them, giving an output of [batch_size x target_len x d_model]
        # using the encoder as the keys and values and the target as the queries

        norm_target_selfattn = 
        norm_encoder_outputs = 
        encdec_attention = 
        # Take the residual between the output and the unnormalized target input of the cross-attention
        res_encdec_attention = 

        output = 

        return output

class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self,
                 embedding_layer, n_layers, n_heads, d_model, d_filter, dropout=None):
        super().__init__()

        self.embedding_layer = embedding_layer
        embed_size = self.embedding_layer.embed_size
        # The encoding stack is a stack of transformer encoder blocks
        self.encoding_stack = []
        for i in range(n_layers):
            encoder = TransformerEncoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self,f"encoder{i}",encoder)
            self.encoding_stack.append(encoder)

    def forward(self, inputs, encoder_mask=None):
        """
            Args:
                inputs: Either a float32 or int32 Tensor with shape [batch_size, sequence_length, ndim]
                encoder_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """

        inputs = self.embedding_layer(inputs)
        output = inputs
        for encoder in self.encoding_stack:
            output = encoder(output, self_attention_mask=encoder_mask)

        return output


class TransformerDecoder(nn.Module):
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    def __init__(self,
                 embedding_layer,
                 output_layer,
                 n_layers,
                 n_heads,
                 d_model,
                 d_filter,
                 dropout = None) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        embed_size = self.embedding_layer.embed_size
        self.decoding_stack = []
        for i in range(n_layers):
            decoder = TransformerDecoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self,f"decoder{i}",decoder)
            self.decoding_stack.append(decoder)
        self.output_layer = output_layer

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def forward(self, target_input, encoder_output, encoder_mask=None, decoder_mask=None, mask_future=False,
        shift_target_sequence_right=False):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                    cache: Used for fast decoding, a dictionary of tf.TensorArray. None during training.
                mask_future: a boolean for whether to mask future states in target self attention

            Returns:
                a tuple of (encoder_output, output)
                    output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        if shift_target_sequence_right:
            target_input = self.shift_target_sequence_right(target_input)

        target_embedding = self.embedding_layer(target_input)

        # Build the future-mask if necessary. This is an upper-triangular mask
        # which is used to prevent the network from attending to later timesteps
        # in the target embedding
        batch_size = target_embedding.shape[0]
        sequence_length = target_embedding.shape[1]
        self_attention_mask = self.get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future)
        # Build the cross-attention mask. This is an upper-left block matrix which takes care of the masking
        # of the output shapes
        cross_attention_mask = self.get_cross_attention_mask(
            encoder_output, target_input, encoder_mask, decoder_mask)

        # Now actually do the decoding which should take us to the right dimension
        decoder_output = target_embedding
        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output, encoder_outputs=encoder_output, self_attention_mask=self_attention_mask, cross_attention_mask=cross_attention_mask)

        # Use the output layer for the final output. For example, this will map to the vocabulary
        output = self.output_layer(decoder_output)
        return output

    def shift_target_sequence_right(self, target_sequence):
        constant_values = 0 if target_sequence.dtype in [th.int32, th.int64] else 1e-10
        pad_array = [1,0,0,0]
        target_sequence = F.pad(target_sequence, pad_array, value=constant_values)[:, :-1]
        return target_sequence

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a Tensor dimension
            :param sequence_length: a Tensor dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask Tensor with shape [batch_size, sequence_length, sequence_length]
        """

        xind = th.arange(sequence_length)[None,:].repeat(*(sequence_length, 1))
        yind = th.arange(sequence_length)[:,None].repeat(*(1, sequence_length))
        mask = yind >= xind
        mask = mask[None,...].repeat(*(batch_size, 1, 1))

        return mask.to(get_device())

    def get_self_attention_mask(self, batch_size, sequence_length, decoder_mask, mask_future):
        if not mask_future:
            return decoder_mask
        elif decoder_mask is None:
            return self.get_future_mask(batch_size, sequence_length)
        else:
            return decoder_mask & self.get_future_mask(batch_size, sequence_length)

    # This is an upper left block matrix which masks the attention for things that don't
    # exist within the internals.
    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
        elif encoder_mask is None:
            # We need to not mask the encoding, but mask the decoding
            # The decoding mask should have shape [batch_size x target_len x target_len]
            # meaning all we have to do is pad the mask out properly
            cross_attention_mask = decoder_mask[:, 1, :][:, None, :].repeat(
                                    *(1, encoder_output.shape[1], 1)).permute((0, 2, 1))
        elif decoder_mask is None:
            cross_attention_mask = encoder_mask[:, 1, :][:, :, None].repeat(
                                    *(1, 1, decoder_input.shape[1])).permute((0, 2, 1))
        else:
            dec_attention_mask = decoder_mask[:, 1, :][:, None, :].repeat(
                                    *(1, encoder_output.shape[1], 1)).permute((0, 2, 1))
            enc_attention_mask = encoder_mask[:, 1, :][:, :, None].repeat(
                                    *(1, 1, decoder_input.shape[1])).permute((0, 2, 1))
            cross_attention_mask = th.logical_and(enc_attention_mask, dec_attention_mask)

        return cross_attention_mask

class TransformerInputEmbedding(nn.Module):

    def __init__(self,
                 embed_size,
                 vocab_size = None,
                 dropout = None,
                 batch_norm = False,
                 embedding_initializer=None) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size) # , weights=[embedding_initializer]

        self.position_encoding = PositionEmbedding(embed_size)
        self.dropout = nn.Dropout(0 if dropout is None else dropout)
        self.batch_norm = None if batch_norm is False else nn.BatchNorm1d(embed_size)

    def forward(self, inputs, start=1):

        # Compute the actual embedding of the inputs by using the embedding layer
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)

        if self.batch_norm:
            embedding = self.batch_norm(embedding.permute((0,2,1))).permute((0,2,1))

        embedding = self.position_encoding(embedding, start=start)
        return embedding

class Transformer(nn.Module):

    def __init__(self,
                 vocab_size = None,
                 n_layers = 6,
                 n_heads = 8,
                 d_model = 512,
                 d_filter = 2048,
                 dropout = None,
                 embedding_initializer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.dropout_weight = 0 if dropout is None else dropout

        input_embedding = TransformerInputEmbedding(d_model, vocab_size, dropout) # , embedding_initializer=embedding_initializer

        output_layer = EmbeddingTranspose(input_embedding.embedding)

        # Build the encoder stack.
        self.encoder = TransformerEncoder(input_embedding, n_layers, n_heads, d_model, d_filter, dropout)

        # Build the decoder stack.
        self.decoder = TransformerDecoder(input_embedding, output_layer, n_layers, n_heads, d_model, d_filter, dropout)

    def forward(self, source_sequence, target_sequence, encoder_mask, decoder_mask, mask_future=True, shift_target_sequence_right=True):

        # Unpack the source and target sequences from the encoder.
        # Source Sequence: [batch_size x source_length]
        # Target Sequence: [batch_size x target_length]
        #
        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = transformer_utils.convert_to_attention_mask(source_sequence, encoder_mask)
        decoder_mask = transformer_utils.convert_to_attention_mask(target_sequence, decoder_mask)

        # After the end of the encoder and decoder generation phase, we have
        # Encoder Mask: [batch_size x source_length x source_length]
        # Decoder Mask: [batch_size x target_length x target_length]

        # Next, we perform the encoding of the sentence. This should take
        # as input a tensor of shape [batch_size x source_length x input_feature_shape]
        # and generate a tensor of shape [batch_size x source_length x d_model]

        ####################################  YOUR CODE HERE  ####################################
        # PART 5: Implement the full Transformer block

        # Using the self.encoder, encode the source_sequence, and provide the encoder_mask variable as the optional mask.
        encoder_output = 

        # Finally, we need to do a decoding this should generate a
        # tensor of shape [batch_size x target_length x d_model]
        # from the encoder output.
        # Using the self.decoder, provide it with the decoder input, and the encoder_output. 
        # As usual, provide it with the encoder and decoder_masks
        # Finally, You should also pass it these two optional arguments:
        # shift_target_sequence_right=shift_target_sequence_right, mask_future=mask_future
        decoder_output = 

        return decoder_output # We return the decoder's output