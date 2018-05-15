
import MySQLdb as mysql
import yaml

import os
import numpy as np
import importlib
import urllib


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value wa
         

         
         
def mysql_query(db,cmd,maxrows=0):
  db.query(cmd)
  r=db.store_result()
  return r.fetch_row(maxrows=maxrows)

         
         
def connect_to_db(db_defs):
  # create connection to the database.
  db = mysql.connect(host=db_defs['ip'],
                    user=db_defs['usrid'],
                    passwd=db_defs['password'],
                    db=db_defs['name'])
         
  return db                  

  
def get_data_count(db,db_defs):
    str ="""SELECT COUNT(*) FROM %s """ % (db_defs['data_table'])
    res = mysql_query(db,str)  
    print('xxxxxxxxxxxx')
    print(res)
    print(res[0][0])
    return res[0][0]
  
def load_dictionary(db,db_defs,n = None):

  if n == None:
    str = ("""SELECT element,code FROM %s""" % (db_defs['dictionary_table']))    
  else:
    str = ("""SELECT element,code FROM %s LIMIT %d""" % (db_defs['dictionary_table'],n))

  data = mysql_query(db,str)  
  reverse_dictionary = dict((y, x) for x, y in data)
  dictionary = dict((x, y) for x, y in data)

  return dictionary, reverse_dictionary

def initialize_db_connection(db_defs):
    '''
    Initialize the database connection and return 
    the dictionaries for the solver...
    '''
    # connect to db 
    db = connect_to_db(db_defs)
    
    #choose db....
    str = ("""USE %s """ % db_defs['name'])
    db.query(str)


    dictionary, reverse_dictionary  = None, None
    if (db_defs['load_dictionary'] == True):
      dictionary, reverse_dictionary = load_dictionary(db,db_defs)

    
    return db, dictionary, reverse_dictionary

    



  
def grab_data_shard(db,db_defs,No,dN):
    '''
    Grabs a section of data from the db.
    
    dN = the max number of points returned
    No = the position in the record to begin drawing points from 
    db = the database connection object
    db_defs -> database definitons -> table names and the likes.
    '''

    str = ("""SELECT var_1,var_2 FROM %s LIMIT %d OFFSET %d""" % (db_defs['data_table'],dN,No))

    # grab the data...
    data = mysql_query(db,str)
    data = np.asarray(data)
    n = len(data)
    target_words = data[:,0]
    context = np.reshape(data[:,1],(-1,1))
    
    print('returning %d of %d points' % (n,dN))
    return target_words, context, n


  
def collect_data(db_defs,number_of_samples = None):
    # connect to db 
    db = connect_to_db(db_defs)
    
    #choose db....
    str = ("""USE %s """ % db_defs['name'])
    db.query(str)

  
    if number_of_samples != None:
      str = ("""SELECT var_1,var_2 FROM %s LIMIT %d""" % (db_defs['data_table'],number_of_samples))
    else:
      str = ("""SELECT var_1,var_2 FROM %s""" % (db_defs['data_table']))
    
    # grab the data...
    data = mysql_query(db,str)
    data = np.asarray(data)
    target_words = data[:,0]
    context = np.reshape(data[:,1],(-1,1))

    dictionary, reverse_dictionary  = None, None
    if (db_defs['load_dictionary'] == True):
      dictionary, reverse_dictionary = load_dictionary(db,db_defs)

    
    return target_words, context, dictionary, reverse_dictionary

    
    
def create_model(model_param,model_specific_params):
    import_string = ('%s' % model_param['model_type'])
    model_def = importlib.import_module(import_string)
    return model_def.Model(model_param,model_specific_params)

    
    
    
def generate_directories(name,id):
    model_dest = os.path.join(name, id,'model')
    log_dir = os.path.join(name, id, 'logs')
    root_dir = os.path.join(name, id)
    return root_dir, model_dest,log_dir
    
      