import pandas as pd
from datasets import plot_dataset
from sklearn.model_selection import train_test_split

def grid_nodes(hidden_min,hidden_max,bottle_min,bottle_max,exp,points=32,exp_times=5):
  h_step = int((hidden_max-hidden_min)/int(sqrt(points)))
  b_step = int((bottle_max-bottle_min)/int(sqrt(points)))
  experiments = []
  for h in range(hidden_min,hidden_max,h_step):
    for b in range(bottle_min,bottle_max,b_step):
      agg = None
      for _ in range(exp_times):
        result = experiment(h,b,exp=exp)
        if agg is None:
          agg = result
        else:
          for k,v in result.items():
            agg[k]+=v # aggregate results for mean
      for k in agg:
        agg[k] /= exp_times
      agg.update({'hidden':h,'bottle':b})
      experiments.append(agg)
  return pd.DataFrame(experiments)

def grid_noise(noise_min,noise_max,exp,points=10,exp_times=5):
  step = int(noise_max-noise_min)
  experiments = []
  for noise in range(int(noise_min*points),int(noise_max*points),step):
    noise = noise/points
    agg = None
    for _ in range(exp_times):
      result = experiment(512,4,exp=exp,noise=noise,corr=1)
      if agg is None:
        agg = result
      else:
        for k,v in result.items():
          agg[k]+=v # aggregate results for mean
    for k in agg:
      agg[k] /= exp_times
    agg.update({'noise':noise})
    experiments.append(agg)
  return pd.DataFrame(experiments)


def experiment(model,train_set,test_set,epochs=300,verbose=False):
  """Possible experiments: mid, rand, alternate and reverse"""
  
  if verbose:
    plot_dataset(train_set)
  model.train(epochs,train_set,verbose)

  
  valid,test = train_test_split(test_set,test_size=0.5)
  if verbose:
    plot_dataset(valid)
    plot_dataset(test)

  before_acc = get_acc(model,df_test)
  
  
  

  train(NN,epochs,df_valid,verbose)
  trained_acc = get_acc(NN,df_test)

  NN = nn.Sequential(OrderedDict([
          ('embeds',nn.Sequential(nn.Linear(3, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,bottleneck),
            nn.ReLU())),
          ('fc',nn.Linear(bottleneck, 2))
          ])  
      )
  for param in NN.embeds.parameters():
      param.requires_grad = False
  train(NN,epochs,df_valid,verbose)
  random_acc = get_acc(NN,df_test)
  return {'before':before_acc,'LLR':trained_acc,'random':random_acc}

