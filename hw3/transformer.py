from typing import Optional, List
from collections import namedtuple

import torch
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
        power = torch.arange(0, hidden_size, step=2, dtype=torch.float32)[:] / hidden_size
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
        # To incorporate that, the authors use a variable frequency sinusoidal embedding.
        # Note that we use zero-indexing here while the authors use one-indexing

        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        batch_size, sequence_length, d_model = inputs.shape

        # obtain a sequence that starts at `start` and increments for `sequence_length`
        seq_pos = torch.arange(start, sequence_length + start, dtype=torch.float32)   # 1-D
        seq_pos_expanded = seq_pos[None,:,None]                                       # dim (1, len, 1)
        index = seq_pos_expanded.repeat(*[1,1,self.hidden_size//2])                   # dim (1, len, h/2) repeat val

        # create the position embedding as described in the paper
        sin_embedding = torch.sin(index / self.divisor)
        cos_embedding = torch.cos(index / self.divisor)

        # interleave the sin and cos. For more info see:
        # https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/3
        position_shape = (1, sequence_length, d_model) # fill in the other two dimensions
        position_embedding = torch.stack((sin_embedding,cos_embedding), dim=3).view(position_shape)

        pos_embed_deviced = position_embedding.to(get_device())
        
        # add the embedding to the input
        output = inputs + pos_embed_deviced
        ####################################  END OF YOUR CODE  ##################################

        return output

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
        #
        # Anton: In the paper, output of each sub-layer is LayerNorm(x+Sublayer(x)). 
        # e.g. output of feedforward is LayerNorm(x + FeedForward(x))        
        # Here we implement LayerNorm(x) + Dropout(FeedForward(x))
        # which isn't correct.
        
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_drop =  self.dropout(dense_out)
        
        # Add the residual here
        output = inputs + dense_drop
        ####################################  END OF YOUR CODE  ##################################
        return output

class TransformerEncoderBlock(nn.Module):
    """An encoding block from the paper Attention Is All You Need."""

    def __init__(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads, [input_size, input_size])
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, input, self_attention_mask=None):
        """Forward call of the encoder block.
        
        Args:
            input: Tensor with shape [batch_size, sequence_length, d_model]

        Returns:
            output: Tensor with same shape as input
        """

        ####################################  YOUR CODE HERE  ####################################
        # PART 4.2: Implement the Transformer Encoder according to section 3.1 of the paper.
        # Perform a multi-headed self-attention across the inputs.
        
        # Anton: again this isn't quite right, LayerNorm used incorrectly.

        # First normalize the input with the LayerNorm initialized in the __init__ function (self.norm)
        norm_inputs = self.norm(input)

        # Apply the self-attention with the normalized input, use the self_attention mask as the optional mask parameter.
        attn = self.self_attention((norm_inputs, norm_inputs), self_attention_mask)

        # Apply the residual connection. res_attn should sum the attention output and the original, non-normalized inputs
        res_attn = input + attn

        # output passes through a feed_forward network
        output = self.feed_forward(res_attn)
        ####################################  END OF YOUR CODE  ##################################
        
        return output


class TransformerDecoderBlock(nn.Module):
    """A decoding block from the paper Attention Is All You Need."""

    def __init__(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout=None) -> None:
        super().__init__()
        
        self.self_norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads, [input_size, input_size])
        self.cross_attention = MultiHeadAttention(n_heads, [input_size, input_size])
        self.cross_norm_encoder = nn.LayerNorm(input_size)
        self.cross_norm_decoder = nn.LayerNorm(input_size)
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, decoder_input, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """Forward call of the decoder block.

        Args:
            decoder_input: a Tensor with shape [batch_size, decoding_sequence_length, channels]
            encoder_output: a Tensor with shape [batch_size, sequence_length, channels]

        Returns:
            output: Tensor with same shape as decoder_inputs
        """
        
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        ####################################  YOUR CODE HERE  ####################################
        # PART 4.2: Implement the Transformer Decoder according to section 3.1 of the paper.

        # layer norm to input of block. 
        norm_decoder_input = self.self_norm(decoder_input)

        # masked self-attention over decoder input
        # output shape [B, decoder_len, d_model]
        decoder_self_attn = self.self_attention((norm_decoder_input, norm_decoder_input), self_attention_mask)
        # residual connection
        res_decoder_self_attn = decoder_self_attn + decoder_input

        # apply respective LayerNorms
        norm_decoder_self_attn = self.cross_norm_decoder(res_decoder_self_attn)
        norm_encoder_output = self.cross_norm_encoder(encoder_output)

        # compute the attention using the k/v from encoder, and q from decoder. 
        cross_attention = self.cross_attention((norm_decoder_self_attn, norm_encoder_output), cross_attention_mask)
        
        # add residual unnormalized decoder input to output of cross-attention
        res_cross_attention = cross_attention + res_decoder_self_attn

        # pass through feedforward
        output = self.feed_forward(res_cross_attention)
        ####################################  END OF YOUR CODE  ##################################
        
        return output

class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self, embedding_layer, n_layers, n_heads, d_model, d_filter, dropout=None):
        super().__init__()

        self.embedding_layer = embedding_layer
        embed_size = self.embedding_layer.embed_size
        # The encoding stack is a stack of transformer encoder blocks
        self.encoding_stack = []
        for i in range(n_layers):
            encoder = TransformerEncoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self, f"encoder{i}", encoder)
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


    def forward(self, target_input, encoder_output, encoder_mask=None, decoder_mask=None, mask_future=False,
        shift_target_sequence_right=False):
        """
        Args:
            target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
            encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
            encoder_mask: ???
            decoder_mask: ???
            mask_future: a boolean for whether to perform masked self-attention
            shift_target_sequence_right: ???

        Returns:
            output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        if shift_target_sequence_right:
            target_input = self.shift_target_sequence_right(target_input)

        target_embedding = self.embedding_layer(target_input)

        # Build the self-attention mask.
        # if mask_future=False, then this is just a padding mask
        # if mask_future=True, then this is a padding mask + upper triangular mask to prevent attending to future targets
        batch_size = target_embedding.shape[0]
        sequence_length = target_embedding.shape[1]
        self_attention_mask = self.get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future)
        
        # Build the cross-attention mask. This is an upper-left block matrix which takes care of the masking
        # of the output shapes
        cross_attention_mask = self.get_cross_attention_mask(encoder_output, target_input, encoder_mask, decoder_mask)

        # Now actually do the decoding which should take us to the right dimension
        decoder_output = target_embedding
        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output, 
                                     encoder_output=encoder_output, 
                                     self_attention_mask=self_attention_mask, 
                                     cross_attention_mask=cross_attention_mask)

        # Use the output layer for the final output. For example, this will map to the vocabulary
        output = self.output_layer(decoder_output)
        return output

    def shift_target_sequence_right(self, target_sequence):
        constant_values = 0 if target_sequence.dtype in [torch.int32, torch.int64] else 1e-10
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

        xind = torch.arange(sequence_length)[None,:].repeat(*(sequence_length, 1))
        yind = torch.arange(sequence_length)[:,None].repeat(*(1, sequence_length))
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
            cross_attention_mask = torch.logical_and(enc_attention_mask, dec_attention_mask)

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
        encoder_output = self.encoder(source_sequence, encoder_mask)

        # Finally, we need to do a decoding this should generate a
        # tensor of shape [batch_size x target_length x d_model]
        # from the encoder output.
        
        # Using the self.decoder, provide it with the decoder input, and the encoder_output. 
        
        # As usual, provide it with the encoder and decoder_masks
        # Finally, You should also pass it these two optional arguments:
        # shift_target_sequence_right=shift_target_sequence_right, mask_future=mask_future
        decoder_output = self.decoder(target_input=target_sequence,
                                      encoder_output=encoder_output,
                                      encoder_mask=encoder_mask,
                                      decoder_mask=decoder_mask,
                                      mask_future=mask_future,
                                      shift_target_sequence_right=shift_target_sequence_right)
        ####################################  END OF YOUR CODE  ##################################

        return decoder_output