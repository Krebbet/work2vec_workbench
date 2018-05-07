#import MySQLdb as mysql
import yaml


# local imports... -> load the 
from solver import Solver
from utils import create_model 
from utils import collect_data
#from utils2 import collect_data2
#from utils2 import create_model
#from utils2 import generate_batch
import numpy as np

from models.constants import *


# extract all parameters for the yml definitions file
PARAM_FILE_DIRECTORY = 'parameters.yml'    

def main():

  # load parameters for run...
  parameters = yaml.load(open(PARAM_FILE_DIRECTORY))
  db_defs = parameters['database']
  solver_param = parameters['solver_param']        
  model_param = parameters['model_param']
  model_specific_params = parameters[model_param['model_type']]

  
  

  print('Collect Data ....')
  #target_words, context, count = collect_data(db_defs,5000)
  data, count, dictionary, reverse_dictionary = collect_data2(vocabulary_size=model_param['vocabulary_size'])
  target_words, context = generate_batch(data, 500, 2, 2)
  
  #data, count, dictionary, reverse_dictionary = collect_data2(vocabulary_size=model_param['vocabulary_size'])
  print('Done Collect Data.')

  print('try to import model')
  # build model...
  model = create_model(model_param,model_specific_params)
  print('Model drawn')
                            
             

  # Initialize the solver object.
  solver = Solver(model)
  
  # train model....
  solver.train(target_words,context,solver_param)
  
  
  #grab embeddings for some sample data.
  embedding = solver.run(np.array([1,2,3,12], dtype=np.int32),solver_param)  
  print('done!')
  
  

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    


                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  