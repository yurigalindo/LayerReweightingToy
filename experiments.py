import pandas as pd
import numpy as np
import torch
from datasets import DataFrameSet, get_dataset
from models import model
from torch.utils.data import DataLoader


def experiment(model: model,train_set: DataFrameSet,test_set: DataFrameSet,optim=None,optim_args=None,
test_split=0.5,pt_epochs=300,epochs=300,verbose=False,batch_size=None):
  """Performs a toy experiment"""
  #TODO: include bias-only model 

  if optim:
    if isinstance(model,torch.nn.Module):
      NN = model
    else:
      NN = model.NN
    optimizer = optim(NN.parameters(),**optim_args)
  else:
    optimizer = None
  if verbose:
    train_set.plot()
  if batch_size:
    train_set = DataLoader(train_set,batch_size,shuffle=True)
  model.train(pt_epochs,train_set,verbose=verbose,optim=optimizer)

  valid,test = test_set.train_test_split(test_size=test_split)
  if verbose:
    valid.plot()
    test.plot()
  if batch_size:
    valid = DataLoader(valid,2*batch_size,shuffle=True)
    test = DataLoader(test,2*batch_size,shuffle=False)
  

  train_acc = model.get_acc(train_set)
  before_acc = model.get_acc(test)
  model.last_layer_reweight()
  model.train(epochs,valid,verbose)
  trained_acc = model.get_acc(test)

  model.reset()
  model.last_layer_reweight()
  model.train(epochs,valid,verbose)
  random_acc = model.get_acc(test)

  return {'Train Acc': train_acc, 'Core-Only':before_acc,'LLR Core-Only':trained_acc,'Random Weights LLR Core-Only':random_acc}

def epoch_tracking(model: model,train_set: DataFrameSet,test_set: DataFrameSet,
test_split=0.5,pt_epochs=300,intervals=10,epochs=300,verbose=False,batch_size=None):
  """Perform a LLR experiment tracking LLR performance over epochs"""
  if isinstance(model,torch.nn.Module):
    optimizer = torch.optim.Adam(model.parameters())
  else:
    # optimizer = torch.optim.Adam(model.NN.parameters())
    # For waterbirds:
    optimizer = torch.optim.SGD(model.NN.parameters(),lr=1e-3,weight_decay=1e-2,momentum=0.9)
  valid,test = test_set.train_test_split(test_size=test_split)
  if batch_size:
    train_set = DataLoader(train_set,batch_size,shuffle=True)
    valid = DataLoader(valid,2*batch_size,shuffle=True)
    test = DataLoader(test,2*batch_size,shuffle=False)
    
  accuracies = []
  for i in range(pt_epochs//intervals):
    model.logistic = None
    train_acc = model.get_acc(train_set)
    before_acc = model.get_acc(test)
    model.last_layer_reweight()
    model.train(epochs,valid,verbose)
    trained_acc = model.get_acc(test)
    accuracies.append({'Train Acc': train_acc, 'Random-Simple':before_acc,'LLR Random-Simple':trained_acc})
    model.logistic = None
    model.train(intervals,train_set,verbose,optim=optimizer)
  
  return pd.DataFrame(accuracies)

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