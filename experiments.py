import pandas as pd
import numpy as np
from datasets import DataFrameSet
from models import model


def experiment(model: model,train_set: DataFrameSet,test_set: DataFrameSet,
test_split=0.5,pt_epochs=300,epochs=300,verbose=False):
  """Performs a toy experiment"""
  
  if verbose:
    train_set.plot()
  model.train(pt_epochs,train_set,verbose)

  valid,test = test_set.train_test_split(test_size=test_split)
  if verbose:
    valid.plot()
    test.plot()

  before_acc = model.get_acc(test)
  model.last_layer_reweight()
  model.train(epochs,valid,verbose)
  trained_acc = model.get_acc(test)

  model.reset()
  model.last_layer_reweight()
  model.train(epochs,valid,verbose)
  random_acc = model.get_acc(test)

  return {'before':before_acc,'LLR':trained_acc,'random':random_acc}

def average_over_exps(args,model,model_args,runs):
  #TODO: epoch with max accuracy
  #TODO: mean and std of differences i.e., LLR-BT, LLR-Rand, LLR-Before
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
    agg = average_over_exps(default_args,model,model_args,exp_times):
    agg.update({arg:x})
    experiments.append(agg)
  return pd.DataFrame(experiments)
