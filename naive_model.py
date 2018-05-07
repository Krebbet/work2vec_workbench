from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import math

#this is a test
from model import Net 
from constants import *


class Model(Net):

  def __init__(self, model_param,specific_param,train_model = True):
    """
    common params: a params dict
    model_params   : a params dict
    """  
    
    self.name = model_param['model_type']
    self.embedding_size = int(model_param['embedding_size'])
    
    # Define all the model variables....
    self.vocabulary_size = int(model_param['vocabulary_size'])
    self.test_size = model_param['test_size']
    self.valid_examples = np.random.choice(model_param['test_window'], model_param['test_size'], replace=False)
    self.batch_norm = model_param['batch_norm']
    
    # build graph!
    if train_model == True:
      self.inference(train_model)
      self.loss()
      self.find_word_similarities()
    else:
      self.inference(train_model)
    
    
    
  def inference(self,train_model):
    # Define our input data placeholders
    self.target_words = tf.placeholder(tf.int32, shape=[None],name='target_words')
    # note: by defining the batch size by this input tensor the
    #   network does not need a batch size parameter, it will shape 
    #   the rest of the tensors to the shape that is fed to this one
    #   through your feeddict.
    self.batch_size = tf.shape(self.target_words)[0] 
    
    #self.x = tf.placeholder(tf.float32,shape=[None,50],name ='X') # input (batch_size * img_size)
    #self.batch_size = tf.shape(self.x)[0]     
    #self.batch_size = tf.shape(self.target_words)

    self.context = tf.placeholder(tf.int32, shape=(None))
    
    # a const set of words to check how the embedding is evolving
    self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
    
    # this propogates whether the model is training or being 
    # used for inference for methods like dropout and batchnorm.
    if train_model == True:
      self.is_training = tf.placeholder(tf.bool,name = 'is_training')
    else :
      self.is_training = tf.constant(False,tf.bool,name = 'is_training')

    # Define our embedding matrix.
    self.embeddings = tf.Variable(
        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        
    # partition out our embedded word vectors that correspond to 
    # the given 'target_words'.
    embed = tf.nn.embedding_lookup(params = self.embeddings, 
                                    ids = self.target_words)


    # apply a linear layer over the entire vocab.
    self.scores = self.linear(embed,
                    output_dim = self.vocabulary_size,
                    is_training = self.is_training,
                    do_batch_norm = self.batch_norm, 
                    init_deviation = 1.0 / math.sqrt(self.embedding_size),
                    reg = REG)
                    

        


  def loss(self):
  
    # convert context K-class data into one-hot vector
    one_hot_context = tf.one_hot(self.context, self.vocabulary_size)


    # get cross - entropy loss ---> this normalizes over entire corpus of words
    soft_max = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=one_hot_context)

    # grab the mean loss as our final loss...
    L_embed = tf.reduce_mean(soft_max)

    # grab our regularization loss 
    L_reg = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))    
    self.loss = L_embed + L_reg


    # collect logging variables
    tf.summary.scalar("Reg_Loss", L_reg)
    tf.summary.scalar("Embedded_Loss", L_embed)
    tf.summary.scalar("Loss", self.loss)    


  def find_word_similarities(self):
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
    normalized_embeddings = self.embeddings / norm

    # grab the 'valid embeddings' for comparison.
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)
    self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)      
