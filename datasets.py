import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



def get_dataset(ds_name,outer_r=2,higher_z=1,**kwargs):
  df = globals()[f"{ds_name}_examples"](1,0,**kwargs)
  df = df.append(globals()[f"{ds_name}_examples"](outer_r,higher_z,**kwargs))
  return DataFrameSet(df)

def simplicity_dataset(linear_n,noisy_n,slab5_n,p_noise=0.2,num_examples=2100):
  num_examples = num_examples//2
  dataset = {}
  # Negative examples
  for i in range(linear_n):
    dataset[f'linear_{i}'] = np.random.rand(num_examples)*0.9 - 1
  for i in range(noisy_n):
    regular = np.random.rand(int(num_examples*(1-p_noise)))*0.9 - 1
    noisy = np.random.rand(int(num_examples*p_noise))*0.2 - 0.1
    total = np.append(regular,noisy)
    np.random.shuffle(total)
    dataset[f'noisy_{i}'] = total
  for i in range(slab5_n):
    first_slab = np.random.rand(num_examples//3)*0.4 - 1
    third_slab = np.random.rand(num_examples//3)*0.4 - 0.2
    fifth_slab = np.random.rand(num_examples//3)*0.4 + 0.6
    dataset[f'slab5_{i}'] = np.append(first_slab,np.append(third_slab,fifth_slab))
  dataset['label'] = np.zeros(num_examples)

  # Positive examples
  for i in range(linear_n):
    total = np.random.rand(num_examples)*0.9 + 0.1
    dataset[f'linear_{i}'] = np.append(dataset[f'linear_{i}'], total)
  for i in range(noisy_n):
    regular = np.random.rand(int(num_examples*(1-p_noise)))*0.9 + 0.1
    noisy = np.random.rand(int(num_examples*p_noise))*0.2 - 0.1
    total = np.append(regular,noisy)
    np.random.shuffle(total)
    dataset[f'noisy_{i}'] = np.append(dataset[f'noisy_{i}'], total)
  for i in range(slab5_n):
    second_slab = np.random.rand(num_examples//2)*0.4 - 0.6
    fourth_slab = np.random.rand(num_examples//2)*0.4 + 0.2
    total = np.append(second_slab,fourth_slab)
    dataset[f'slab5_{i}'] = np.append(dataset[f'slab5_{i}'], total)
  dataset['label'] = np.append(dataset['label'], np.ones(num_examples))
  dataset = pd.DataFrame(dataset)
  return DataFrameSet(dataset)

def train_examples(R,label,noise,corr=1,num_examples=2000):
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
    self.df = df
    self.x = torch.tensor(df.iloc[:,:-1].values,dtype=torch.float32)
    self.y = torch.tensor(df['label'].values,dtype=torch.float32)
  def __len__(self):
    return len(self.y)
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def plot(self):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(self.df.iloc[:,0],self.df.iloc[:,1],self.df.iloc[:,2],c=self.df['label'])
    fig.show()
  def train_test_split(self,**kwargs):
    X_train, X_test = train_test_split(self.df,**kwargs)
    return DataFrameSet(X_train),DataFrameSet(X_test)

class SimplicitySet(Dataset):
  def __init__(self,df):
    self.df = {}
    self.df['X'] = df['X'].detach().numpy()
    self.df['Y'] = df['Y'].detach().numpy()
    self.x = df['X']
    self.y = df['Y']
  def __len__(self):
    return len(self.y)
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def plot(self):
    print("Cant plot simplicity bias dataset")
  def train_test_split(self,**kwargs):
    X_train, X_test = train_test_split(self.df['X'],**kwargs)
    Y_train, Y_test = train_test_split(self.df['Y'],**kwargs)
    return SimplicitySet({'X':torch.tensor(X_train,dtype=torch.float32),
      'Y':torch.tensor(Y_train,dtype=torch.float32)}), SimplicitySet(
        {'X':torch.tensor(X_test,dtype=torch.float32),
      'Y':torch.tensor(Y_test,dtype=torch.float32)})


