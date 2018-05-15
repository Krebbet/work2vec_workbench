'''
This solver unit is to simplify the model 
code and to reduce repeated code...

it is designed to take in a model and data and train the thing.
'''

import os
import numpy as np
import math
import io
import tensorflow as tf

# utils imports
from utils import generate_directories
from utils import get_data_count
from utils import grab_data_shard


class Solver(object):
  """
   This creates a solver architecture to
   train a word2vec model.
   
   It has two primary functions:
    train -> train a given model with a given dataset
    run   -> extract a list of learned word embeddings from 
      the model.
      
      
    The solver takes target_word data, context data as 
    word pairings. These should be provided as numpy arrays.
  """
  
  def __init__(self,model):
    self.m = model


   
      
  def optimize(self,learning_rate,var_list = None):    
    ## OPTIMIZER ## 
    
    model = self.m
    with tf.name_scope('Optimize'):
      # the adam optimizer is much more stable and optimizes quicker than
      # basic gradient descent. (*Ask me for details*)
      optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
      
      # This is generally for fine-tuning,
      # Instead of optimizing all variables you can 
      # choose a set to train.
      if (var_list == None):
        grads=optimizer.compute_gradients(model.loss)
      else:
        grads=optimizer.compute_gradients(model.loss,var_list = var_list)
        
      # we cap the gradient sizes to make sure we do not
      # get outlier large grads from computational small number errors
      with tf.name_scope('Clip_Grads'):
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
                
        self.train_op = optimizer.apply_gradients(grads)
          
    return         

    
    
  def run(self,target_words,param):
    '''
      Returns a numpy matrix holding the 
      learned embeddings from the network defined
      in params.
      
      return shape ->  [target words, embedding size]
      
      
      
    '''
    # define params locally for readability
    model = self.m     
    batch_size = param['batch_size']
    id = param['id']
    
    # set up all the directories to send the model and logs...
    root_dir, model_dest,_ = generate_directories(model.name,id)

    # set the training feed.
    fetches=[]
    fetches.extend([model.embed])

    # determine the counter values.
    n = target_words.shape[0]
 
    itts = max(n // batch_size, 1)


    embedding = None #numpy.empty()
    with tf.Session() as sess:
      # setup a saver object to save model...
      # create model saver
      saver = tf.train.Saver()
      
      # initialize variables -> need to add code to load if loading trained model.
      if os.path.exists(root_dir):
        print('loaded model!')
        saver.restore(sess, model_dest) 
      else:  
        print('initialized model!')
        tf.global_variables_initializer().run()
  
      for i in range(itts):
        if (i == itts - 1):
          tw_batch = target_words[batch_size*i:]
        else:  
          tw_batch = target_words[batch_size*i:batch_size*(i +1)]
        


        feed_dict={model.target_words:tw_batch,
                    model.is_training:False}   
                    
        [embedding_batch]=sess.run([model.embed],feed_dict)
        if embedding == None:
          embedding = embedding_batch
        else :
          embedding = np.hstack((embedding, embedding_batch))
        

    return embedding        
  
  
  #def train(self,target_words,context,dictionary, reverse_dictionary,param):
  def train(self,db,db_defs,dictionary, reverse_dictionary,param):
    '''
    This routine takes in the database connection to the 
    training data target_words and 
    context pairing and trains the word embedding, using the 
    model architecture fed into the solver object.
    
    The model variables will be initialized as specified, unless
    a previous model has been saved to the models destination directory.
    
    All logs and model data will be saved and loaded from
     './model.name/id/' as defined in the parameters.
     
     
    All hyper parameters can be adjusted in paramters.yml. 
    
    '''
  
    # define params locally for readability
    model = self.m     
    learning_rate = param['learning_rate']
    epoch = param['epoch']
    batch_size = param['batch_size']
    id = param['id']
    top_k = param['print_top_k']
    shard_size = param['shard_size']

    
    # set up all the directories to send the model and logs...
    _ , model_dest,log_dir = generate_directories(model.name,id)

   
  
    # get training op.. set learning rate...
    self.optimize(learning_rate)


    # set the training feed.
    fetches=[]
    fetches.extend([self.train_op])

    # determine the counter values.
    num_train = get_data_count(db,db_defs)

    iterations_per_shard = max(shard_size // batch_size, 1) 
    if (shard_size % batch_size) != 0 :
      iterations_per_shard += 1

    shards_per_epoch = max(num_train // shard_size, 1)
    iterations_per_epoch = iterations_per_shard*shards_per_epoch   \
      + (num_train % shard_size) // batch_size 


    num_iterations = epoch * iterations_per_epoch

    print('***********************************************')
    print('Begining training of %s model is id: %s' % (model.name,id))    
    print('Training Points: %d' % (num_train))
    print('Batch size = %d' % (batch_size))
    print('Shard Size: %d (iters/shard): %d' % (shard_size,iterations_per_shard))
    print('Epoch: %d  shards/epoch: %d iters/epoch: %d' % (epoch,shards_per_epoch,iterations_per_epoch))
    print('Total Iterations: %d' % (num_iterations))
    print('model will be saved to: %s' % model_dest)
    print('logs will be stored in: %s' % log_dir)

    
    
    with tf.Session() as sess:

      # setup a saver object to save model...
      # create model saver
      saver = tf.train.Saver()
      
    
      # initialize variables -> need to add code to load if loading trained model.
      if os.path.exists(model_dest):
        saver.restore(sess, model_dest) 
      else:  
        tf.global_variables_initializer().run()
  
      # create session writers
      writer = tf.summary.FileWriter(log_dir,sess.graph) # for 1.0
      #test_writer = tf.summary.FileWriter(os.path.join(test_dir , model.model_name)) # for 1.0
      merged = tf.summary.merge_all()
      
      
      
      print('Begin Training')
      print('***********************************************')
      
      for e in range(epoch):
        # create a mask to shuffle the data
        # uniquely each epoch.
        
        # draw data from db in controllable shards.
        No = 0
        n = dN
        shard_n = 0

        while (n == dN):
          
          
          target_words, context, n = grab_data_shard(db,db_defs,No,dN)
          
          # make sure we do not put in an empty data set
          if n == 0:
            break
          
          No += dN
          shard_n += 1

          
          mask = np.arange(n)
          np.random.shuffle(mask)
          
          

          for i in range(iterations_per_epoch):
            
            # print out tracking information to make sure everything is running correctly...
            if ( (i + iterations_per_shard*(shard_n -1)) % param['read_out'] == 0):
              print('%d of %d for shard %d' % (i , iterations_per_shard,shard_n))
              print('%d of %d for epoch %d' % (i + iterations_per_shard*(shard_n -1),iterations_per_epoch,e))
            
            # Grab the batch data... (handle modified batches...)
            if batch_size*(i + 1) > len(target_words):
              target_batch = target_words[mask[batch_size*i:]]
              context_batch = context[mask[batch_size*i:]]            
            else :
              target_batch = target_words[mask[batch_size*i:batch_size*(i +1)]]
              context_batch = context[mask[batch_size*i:batch_size*(i +1)]]
            

            feed_dict={model.target_words:target_batch,
                       model.context:context_batch,
                       model.is_training:True}      
            

            # do training on batch, return the summary and any results...
            [summary,_]=sess.run([merged,fetches],feed_dict)

            
            # write summary to writer
            writer.add_summary(summary, i + e*iterations_per_epoch)
          
          
          # epoch done, check word similarities....
          # Note: I do not have access to the word library so 
          # I cannot create my own reverse lookup...
          # we can set this up pretty easy tho.
          if (e % param['similarity_readout'] == 0):
            [sim] = sess.run([model.similarity])
            for i in range(model.test_size):
              #valid_word = reverse_dictionary[model.valid_examples[i]]
              valid_word = model.valid_examples[i]
              #x = -sim[i, :].argsort()
              nearest = (-sim[i, :]).argsort()[1:top_k + 1]
              log_str = 'Nearest to %s:' % valid_word
              for k in range(top_k):
                #close_word = reverse_dictionary[nearest[k]]
                close_word = nearest[k]
                if reverse_dictionary == None:
                  log_str = '%s %d,' % (log_str, close_word)
                else: 
                  log_str = '%s %s,' % (log_str, reverse_dictionary[close_word])
              print(log_str)        
        
          
          # checkpoint the model while training... 
          if (e % param['check_point'] == 0):
            saver.save(sess,model_dest, global_step=e+1)
          print('%d of %d epoch complete.' % (1+e,epoch))
          
          
  
      ## TRAINING FINISHED ##
      # saves variables learned during training
      saver.save(sess,model_dest)  
      
      # make sure the log writer closes and sess is done.
      writer.close()
      sess.close()  


    print('***********************************************')
    print('Done training')
    print('model saved to: %s' % model_dest)
    print('logs stored in: %s' % log_dir)
    print('***********************************************')
    return