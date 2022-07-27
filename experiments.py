import pandas as pd
from datasets import DataFrameSet
from models import model


def experiment(model: model,train_set: DataFrameSet,test_set,pt_epochs=300,epochs=300,verbose=False):
  """Performs a toy experiment"""
  
  if verbose:
    train_set.plot()
  model.train(pt_epochs,train_set,verbose)

  valid,test = test_set.train_test_split(test_size=0.5)
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

def grid_kwarg(x_min,x_max,arg,default_args,arg_type="int",points=10,exp_times=5):
  step = int(x_max-x_min)
  experiments = []
  for x in range(int(x_min*points),int(x_max*points),step):
    x = x/points
    if arg_type == "int":
      x = int(x)
    agg = None
    for _ in range(exp_times):
      default_args[arg]=x
      result = experiment(**default_args)
      if agg is None:
        agg = result
      else:
        for k,v in result.items():
          agg[k]+=v # aggregate results for mean
    for k in agg:
      agg[k] /= exp_times
    agg.update({arg:x})
    experiments.append(agg)
  return pd.DataFrame(experiments)

def grid_kwarg(x_min,x_max,arg,default_args,arg_type="int",points=10,exp_times=5):
  step = int(x_max-x_min)
  experiments = []
  for x in range(int(x_min*points),int(x_max*points),step):
    x = x/points
    if arg_type == "int":
      x = int(x)
    agg = None
    for _ in range(exp_times):
      default_args[arg]=x
      result = experiment(**default_args)
      if agg is None:
        agg = result
      else:
        for k,v in result.items():
          agg[k]+=v # aggregate results for mean
    for k in agg:
      agg[k] /= exp_times
    agg.update({arg:x})
    experiments.append(agg)
  return pd.DataFrame(experiments)