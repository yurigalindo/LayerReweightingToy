import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import nn
from collections import OrderedDict
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt

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


def experiment(x_t=1,y_t=1,noise_radius=1,hidden_units=512,bottleneck=16,epochs=300,verbose=False,noise=0):
  """Possible experiments: mid, rand, alternate and reverse"""
  df = train_label_examples(1,0,noise,noise_radius)
  df = df.append(train_label_examples(2,1,noise,noise_radius))
  if verbose:
    plt.scatter(df['x'],df['y'],c=df['label'])
    plt.show()
  NN = nn.Sequential(OrderedDict([
          ('embeds',nn.Sequential(nn.Linear(2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,bottleneck),
            nn.ReLU())),
          ('fc',nn.Linear(bottleneck, 2))
          ])  
      )

  DF = DataFrameSet(df)
  train(NN,epochs,DF,verbose)
  df2 = test_examples(1,0,x_t,y_t)
  df2 = df2.append(test_examples(2,1,x_t,y_t))
  
  valid,test = train_test_split(df2,test_size=0.5)
  if verbose:
    plt.scatter(valid['x'],valid['y'],c=valid['label'])
    plt.show()
    plt.scatter(test['x'],test['y'],c=test['label'])
    plt.show()
  df_valid = DataFrameSet(valid)
  df_test = DataFrameSet(test)

  before_acc = get_acc(NN,df_test)
  for param in NN.parameters():
      param.requires_grad = False
  NN.fc = nn.Linear(bottleneck, 2)
  
  

  train(NN,epochs,df_valid,verbose)
  trained_acc = get_acc(NN,df_test)

  NN = nn.Sequential(OrderedDict([
          ('embeds',nn.Sequential(nn.Linear(2, hidden_units),
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

def train_label_examples(R,label,noise=0,noise_radius=1,num_examples=2000):
  """Returns a Dataframe with training examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  noise_portion = int(noise*num_examples/3) # how many examples will be affected by each radius
  x_radius = noise_radius*np.random.randn(3) # x to translate on affected
  x_noises = [np.ones(noise_portion)*x_r for x_r in x_radius]
  x_noises.append(np.zeros(num_examples-3*noise_portion))
  #print(x_noises)
  x_total = []
  for x_n in x_noises:
    x_total.extend(x_n)
  np.random.shuffle(x_total)
  
  y_radius = noise_radius*np.random.randn(3)
  y_noises = [np.ones(noise_portion)*y_r for y_r in y_radius]
  y_noises.append(np.zeros(num_examples-3*noise_portion))
  y_total = []
  for y_n in y_noises:
    y_total.extend(y_n)
  np.random.shuffle(y_total)

  xnums = np.append(xnums,xnums) + np.array(x_total)
  ynums = np.append(ynums,-ynums) + np.array(y_total)
  labels = label*np.ones_like(xnums)  
  return pd.DataFrame({'x':xnums,'y':ynums,'label':labels})

def test_examples(R,label,x_t=1,y_t=1,noise=0,num_examples=2000):
  """Returns a Dataframe with testing examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  xnums = np.append(xnums,xnums) + noise*np.random.randn(num_examples)
  xnums += x_t
  ynums = np.append(ynums,-ynums) + noise*np.random.randn(num_examples)
  ynums += y_t
  labels = label*np.ones_like(xnums)
  return pd.DataFrame({'x':xnums,'y':ynums,'label':labels})

def mid_examples(R,label,num_examples=4000):
  """Returns a Dataframe with testing examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  xnums = np.append(xnums,xnums)
  ynums = np.append(ynums,-ynums)
  labels = label*np.ones_like(xnums)
  znums = 0.5*np.ones_like(xnums)
  return pd.DataFrame({'x':xnums,'y':ynums,'z':znums,'label':labels})

def reverse_examples(R,label,num_examples=4000):
  """Returns a Dataframe with testing examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  xnums = np.append(xnums,xnums)
  ynums = np.append(ynums,-ynums)
  labels = label*np.ones_like(xnums)
  znums = (1-label)*np.ones_like(xnums)
  return pd.DataFrame({'x':xnums,'y':ynums,'z':znums,'label':labels})

def rand_examples(R,label,num_examples=4000):
  """Returns a Dataframe with testing examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  xnums = np.append(xnums,xnums)
  ynums = np.append(ynums,-ynums)
  labels = label*np.ones_like(xnums)
  znums = np.random.rand(num_examples)
  return pd.DataFrame({'x':xnums,'y':ynums,'z':znums,'label':labels})

def alternate_examples(R,label,num_examples=4000):
  """Returns a Dataframe with testing examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  xnums = np.append(xnums,xnums)
  ynums = np.append(ynums,-ynums)
  labels = label*np.ones_like(xnums)
  znums = np.random.randint(low=0,high=2,size=num_examples)
  return pd.DataFrame({'x':xnums,'y':ynums,'z':znums,'label':labels})

class DataFrameSet(Dataset):
  def __init__(self,df):
    self.x = torch.tensor(df[['x','y']].values,dtype=torch.float32)
    self.y = torch.tensor(df['label'].values,dtype=torch.float32)
  def __len__(self):
    return len(self.y)
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  

def train(NN,epochs,dataset,verbose):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(NN.parameters())
  losses = []
  accs = []
  for i in range(epochs):
    x,y=dataset[:]
    optimizer.zero_grad()

    preds = NN(x.float())
    #print(preds)
    loss = criterion(preds,y.long())
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    predictions = torch.argmax(preds,dim=1)
    correct = (predictions == y).float().sum()
    accs.append(correct/len(y))
  if verbose:
    plt.plot(losses)
    plt.plot(accs)
    plt.show()

def get_acc(NN,dataset):
  with torch.no_grad():
    x,y = dataset[:]
    preds = NN(x.float())
    predictions = torch.argmax(preds,dim=1)
    correct = (predictions == y).float().sum()
    return correct/len(y)


