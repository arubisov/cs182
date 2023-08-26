import torch as th
from torch import nn
from torch.nn import functional as F

class AttentionLayerLoss(nn.Module):
  """
  Computes the difference between teacher and student attention output.
  Attention weights learned by large language models hold sophisticated
  syntax information, so are worth matching on their own.
  """
  def __init__(self):
    super().__init__()
  def forward(self,teacher_attn,student_attn):
    """MSE loss of teacher attention and projected student attention.

        :param teacher_attn: Tensor with shape [batch_size, channels]
        :param student_attn: Tensor with shape [batch_size, channels]

        :return: loss: Tensor with shape [1,]
    """
    ####################################  YOUR CODE HERE  ####################################
    # PART 1: Implement Attention Layer Loss
    # Take the MSE loss between the student and teacher attention.
    # To simplify calculations, remember that E[E[X|Y]] = E[X].
    teacher_attn.detach_()
    loss = 
    return loss
    ####################################  END OF YOUR CODE  ##################################

class HiddenLayerLoss(nn.Module):
    """
    Computes the difference between teacher and student encoder block output
    """
    def __init__(self,teacher_dim,student_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim,teacher_dim)
        
    def forward(self,teacher_hddn,student_hddn):
        """MSE loss of teacher's encoder block and projected student output.

            :param teacher_hddn: Tensor with shape [batch_size, teacher_hidden_size]
            :param student_hddn: Tensor with shape [batch_size, student_hidden_size]

            :return: loss: Tensor with shape [1,]
        """
        ####################################  YOUR CODE HERE  ####################################
        # PART 2: Implement Hidden Layer Loss
        # Project the student output into the same space as the teacher output.
        # then take the MSE loss between the student and teacher hidden layer outputs.
        # To simplify calculations, remember that E[E[X|Y]] = E[X].
        teacher_hddn.detach_()
        proj_student = 
        loss = 
        return loss
        ####################################  END OF YOUR CODE  ##################################

class EmbeddingLayerLoss(nn.Module):
    """
    Computes the difference between teacher and student embedding output
    """
    def __init__(self,teacher_dim,student_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim,teacher_dim)
    def forward(self,teacher_embd,student_embd):
        """MSE loss of teacher embeddings and projected student embeddings.

            :param teacher_embd: Tensor with shape [word_count, teacher_embedding_size]
            :param student_embd: Tensor with shape [word_count, student_embedding_size]

            :return: loss: Tensor with shape [1,]
        """
        ####################################  YOUR CODE HERE  ####################################
        # PART 3: Implement Embedding Layer Loss
        # Project the student embedding into the same space as the teacher embedding.
        # Then take their MSE loss.
        # To simplify calculations, remember that E[E[X|Y]] = E[X].
        teacher_embd.detach_()
        proj_student = 
        loss = 
        return loss
        ####################################  END OF YOUR CODE  ##################################
    
class PredictionLoss(nn.Module):
    """
    Computes the difference between teacher and student predicted logits
    """
    def __init__(self):
        super().__init__()
    def forward(self,teacher_pred,student_pred,t=1):
        """Soft Cross-entropy loss of teacher and student prediction logits.

            :param teacher_pred: Tensor with shape [batch_size, word_count]
            :param student_pred: Tensor with shape [batch_size, word_count]

            :return: loss: Tensor with shape [1,]
        """
        ####################################  YOUR CODE HERE  ####################################
        # PART 4: Implement Prediction Layer Loss
        # Take the soft cross-entropy loss bewteen the teacher and student logits.
        # Keep in mind that the cross entropy term `p_target*log(p_prediction)` is asymmetrical between the target and prediction
        # The F.softmax and F.log_softmax will be helpful here
        # Also keep in mind that the last dimension of the prediction is the feature dimension.
        teacher_pred.detach_()
        target_terms = 
        pred_terms = 
        samplewise_sce = 
        mean_sce = samplewise_sce.mean()
        return mean_sce
        ####################################  END OF YOUR CODE  ##################################

class KnowledgeDistillationLoss(nn.Module):
    """
    Computes the total difference between all layer outputs of student and teacher,
    following a mapping between teacher and student encoder blocks.
    """
    def __init__(self,teacher_embd_dim,student_embd_dim,teacher_hddn_dim,student_hddn_dim,layer_mapping):
        super().__init__()
        self.layer_mapping = layer_mapping
        self.embedding_loss = EmbeddingLayerLoss(teacher_embd_dim,student_embd_dim)
        for i in range(len(layer_mapping)):
            attention_loss = AttentionLayerLoss()
            self.__setattr__(f"attention_loss{i}",attention_loss)
            
            hidden_loss = HiddenLayerLoss(teacher_hddn_dim,student_hddn_dim)
            self.__setattr__(f"hidden_loss{i}",hidden_loss)
        self.prediction_loss = PredictionLoss()

    def forward(self,teacher_out,student_out,penalize_prediction=False):
        """Soft Cross-entropy loss of teacher and student prediction logits.

            :param teacher_out: Dictionary {
              embeddings : the entire embedding block of the model,
              attentions : the attention outputs at each layer for each sample,
              hidden_states : the output of each encoder block for each sample,
              logits : the prediction logits for each sample.
            }
            :param student_embd: Dictionary with same structure as teacher_out

            :return: loss: Tensor with shape [1,]
        """
        ####################################  YOUR CODE HERE  ####################################
        # The total loss in comparing the teacher and student models is the sum of losses for each module.
        # Since the student will likely have less intermediate encoder blocks than the teacher,
        # there needs to be a mapping between teacher and student blocks. 
        
        # take the loss for the embedding
        embedding_loss = self.embedding_loss(teacher_out['embeddings'],student_out['embeddings'])
        
        # apply the loss from each attention and hidden layer based on the layer mapping
        attention_loss = 0
        hidden_loss = 0
        for st_i,te_i in enumerate(self.layer_mapping):
            attn_fn = self.__getattr__(f"attention_loss{st_i}")
            attention_loss += 
            hddn_fn = self.__getattr__(f"hidden_loss{st_i}")
            hidden_loss += 
            
        # sum up the loss for each layer
        loss = 
        
        # apply the prediction penalty during task distillation
        if penalize_prediction:
            prediction_loss = 
            loss += 
        return loss
        ####################################  END OF YOUR CODE  ##################################
        