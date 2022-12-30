
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from models import bottle_logistic
from sklearn.utils import shuffle


# Experiments

def track_datapoints_experiment(epochs,train_set,in_set,core_set,random_set,model_args,frequency=10):
  model = bottle_logistic(**model_args)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.NN.parameters())

  noisy_accs = {'in_distribution':[],"core-only":[],"random-simple LLR":[]}
  clean_accs = {'in_distribution':[],"core-only":[],"random-simple LLR":[]}
  for i in range(epochs//frequency):
    last_in = None
    last_core = None
    last_llr = None
    for x,y,data_type in train_set:
        # train on this datapoint
        optimizer.zero_grad()
        preds = model.NN(x.float())
        loss = criterion(preds,y.long())
        loss.backward()
        optimizer.step()

        type_acc =  [clean_accs,noisy_accs][int(data_type)]

        # get accs
        in_acc = model.get_acc(in_set)
        core_acc = model.get_acc(core_set)
        if last_in:
            type_acc['in_distribution'].append(in_acc - last_in)
            type_acc['core-only'].append(core_acc - last_core)
        last_in = in_acc
        last_core = core_acc

        # do LLR
        valid,test = random_set.train_test_split()
        model.last_layer_reweight()
        model.train(None,valid,False)
        llr_acc = model.get_acc(test)
        if last_llr:
            type_acc['random-simple LLR'].append(llr_acc - last_llr)
        last_llr = llr_acc

        # undo LLR
        for param in model.NN.embeds.parameters():
            param.requires_grad = True
        model.logistic = None
        model.scaler = None
        
    model.train(frequency-1,in_set,False) #TODO: Actually use the train set for this step

  for k,v in clean_accs.items():
    array = np.array(v)
    clean_accs[k] = array
    plt.title(f"clean - {k}")
    plt.plot(array,label = k)
    plt.show()
  
  for k,v in noisy_accs.items():
    array =  np.array(v)
    noisy_accs[k] = array
    plt.title(f"noisy - {k}")
    plt.plot(array,label= k)
    plt.show()

  return clean_accs,noisy_accs



# Datasets

def type_slab(p_noise=0.2,noise_size=0.2,num_examples=2100):
  num_examples = num_examples//2
  regular_size = int(num_examples*(1-p_noise))
  a_third = num_examples//3
  dataset = {}
  # Negative examples
  regular = np.random.rand(regular_size)*(1-noise_size/2) - 1
  regular = np.vstack((regular,np.zeros(regular_size)))
  noisy = np.random.rand(num_examples - regular_size)*noise_size - noise_size/2
  noisy = np.vstack((noisy,np.ones(num_examples -regular_size)))
  total = np.hstack((regular,noisy))
  total = np.transpose(total)
  np.random.shuffle(total)
  dataset['noisy'] = total[:,0]
  dataset['type'] = total[:,1]

  first_slab = np.random.rand(a_third)*0.4 - 1
  third_slab = np.random.rand(a_third)*0.4 - 0.2
  fifth_slab = np.random.rand(num_examples-2*a_third)*0.4 + 0.6
  dataset['slab'] = np.append(first_slab,np.append(third_slab,fifth_slab))
  dataset['label'] = np.zeros(num_examples)

  # Positive examples
  regular = np.random.rand(regular_size)*(1-noise_size/2) + noise_size/2
  regular = np.vstack((regular,np.zeros(regular_size)))
  noisy = np.random.rand(num_examples - regular_size)*noise_size - noise_size/2
  noisy = np.vstack((noisy,np.ones(num_examples - regular_size)))
  total = np.hstack((regular,noisy))
  total = np.transpose(total)
  np.random.shuffle(total)
  dataset['noisy'] = np.append(dataset['noisy'], total[:,0])
  dataset['type'] = np.append(dataset['type'],total[:,1])
  second_slab = np.random.rand(num_examples//2)*0.4 - 0.6
  fourth_slab = np.random.rand(num_examples -num_examples//2)*0.4 + 0.2
  total_slabs = np.append(second_slab,fourth_slab)
  dataset['slab'] = np.append(dataset['slab'], total_slabs)
  dataset['label'] = np.append(dataset['label'], np.ones(num_examples))
  
  dataset = pd.DataFrame(dataset)
  dataset = shuffle(dataset)
  return TypeSet(dataset)

class TypeSet(Dataset):
  """Dataset structure for point tracking experiment
  """
  def __init__(self,df):
    self.df = df
    self.x = torch.tensor(df[['noisy','slab']].values,dtype=torch.float32)
    self.y = torch.tensor(df['label'].values,dtype=torch.float32)
    self.type = df['type']
  def __len__(self):
    return len(self.y)
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx],self.type[idx]
  def plot(self):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(self.df['noisy'],self.df['slab'],c=self.df['label'])
    fig.show()

# Aux functions

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n