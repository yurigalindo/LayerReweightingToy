import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

def plot_dataset(df):
    # TODO: test
    df = df[:][0]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['x'],df['y'],df['z'],c=df['label'])
    fig.show()

def get_dataset(ds_name,**kwargs):
    df = globals()[f"{ds_name}_examples"](1,0,**kwargs)
    df = df.append(globals()[f"{ds_name}_examples"](2,1,**kwargs))
    return DataFrameSet(df)

def train_examples(R,label,corr,noise,num_examples):
  """Returns a Dataframe with testing examples of one label, in the given radius
  """
  xnums = np.linspace(-R,R,num_examples//2)
  ynums = np.sqrt(R**2 - np.square(xnums))
  xnums = np.append(xnums,xnums)
  ynums = np.append(ynums,-ynums)
  labels = label*np.ones_like(xnums)
  znums = np.copy(labels)
  idx_to_flip = random.sample(range(num_examples), int((1-corr)*num_examples))
  znums[idx_to_flip] = (1-label)
  znums += noise*(np.random.rand(num_examples) - 0.5) # 0 centered and noise norm
  
  return pd.DataFrame({'x':xnums,'y':ynums,'z':znums,'label':labels})

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
    self.x = torch.tensor(df[['x','y','z']].values,dtype=torch.float32)
    self.y = torch.tensor(df['label'].values,dtype=torch.float32)
  def __len__(self):
    return len(self.y)
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  

