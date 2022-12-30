import pandas as pd
import numpy as np
from datasets import DataFrameSet, get_dataset
from models import model


def experiment(model: model,train_set: DataFrameSet,test_set: DataFrameSet,
test_split=0.5,pt_epochs=300,epochs=300,verbose=False,features=False):
  """Performs a toy experiment"""
  #TODO: include bias-only model 
  
  if verbose:
    train_set.plot()
  model.train(pt_epochs,train_set,verbose,False)

  valid,test = test_set.train_test_split(test_size=test_split)
  if verbose:
    valid.plot()
    test.plot()

  if features:
    model.contour_features()
  before_acc = model.get_acc(test)
  model.last_layer_reweight()
  model.train(epochs,valid,verbose)
  trained_acc = model.get_acc(test)

  model.reset()
  model.last_layer_reweight()
  model.train(epochs,valid,verbose)
  random_acc = model.get_acc(test)

  return {'Core-Only':before_acc,'LLR Core-Only':trained_acc,'Random Weights LLR Core-Only':random_acc}

def average_over_exps(args,model,model_args,runs):
  #TODO: epoch with max accuracy
  
  agg = {}
  for _ in range(runs):
    args['model']=model(**model_args)
    result = experiment(**args)   
    if not agg:
      for k,v in result.items():
        agg[k]=[v] # start the aggreation
    else:
      for k,v in result.items():
        agg[k].append(v) # add following results
  result = {}
  for k,v in agg.items():
    result[f"{k}_mean"] = np.mean(v)
    result[f"{k}_std"] = np.std(v)
  return result

def grid_kwarg(x_min,x_max,arg,default_args,model,model_args,arg_type="int",points=10,exp_times=5):
  experiments = []
  for x in np.linspace(x_min, x_max, num=points,endpoint=False):
    if arg_type == "int":
      x = int(x)
    default_args[arg]=x
    agg = average_over_exps(default_args,model,model_args,exp_times)
    agg.update({arg:x})
    experiments.append(agg)
  return pd.DataFrame(experiments)

  #TODO: Grid over dataset

def grid_model(x_min,x_max,arg,default_args,model,model_args,arg_type="int",points=10,exp_times=5):
  experiments = []
  for x in np.linspace(x_min, x_max, num=points,endpoint=False):
    if arg_type == "int":
      x = int(x)
    model_args[arg]=x
    agg = average_over_exps(default_args,model,model_args,exp_times)
    agg.update({arg:x})
    experiments.append(agg)
  return pd.DataFrame(experiments)


def random_search(list_dict_args,list_dict_model_args,default_args,default_model_args,model,points=30,exp_times=3):
  """ Expects lists of dicts with keys arg, min, max, type, and sample"""
  experiments = []
  for _ in range(points):
    for arg in list_dict_args:
      if arg["sample"]=="uniform":
        x = np.random.uniform(arg["min"],arg["max"])
      else:
        x = np.random.uniform(np.log(arg["min"]),np.log(arg["max"]))
        x = np.exp(x)
      if arg["type"] == "int":
        x = int(x)
      default_args[arg["arg"]]=x
    for arg in list_dict_model_args:
      if arg["sample"]=="uniform":
        x = np.random.uniform(arg["min"],arg["max"])
      else:
        x = np.random.uniform(np.log(arg["min"]),np.log(arg["max"]))
        x = np.exp(x)
      if arg["type"] == "int":
        x = int(x)
      default_model_args[arg["arg"]]=x    
    agg = average_over_exps(default_args,model,default_model_args,exp_times)
    agg.update(default_args)
    agg.update(default_model_args)
    experiments.append(agg)
  return pd.DataFrame(experiments)