
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def track_gradients_experiment(epochs,clean_set,noisy_set,in_loader,random_loader,model,frequency=10):
  optimizer = torch.optim.Adam(model.parameters())
  loss_f = nn.CrossEntropyLoss()

  global grad
  grad = 0 # global variable using for storing grad of pool layer
  def update_grad(x,y,z):
    global grad
    grad = torch.squeeze(z[0])
  model.pool.register_backward_hook(update_grad) # registering hook for storing grad

  global output
  output = 0 # global variable using for storing output of pool layer
  def update_output(x,y,z):
    global output
    output = torch.squeeze(z[0])
  model.pool.register_forward_hook(update_output) # registering hook for storing output

  clean_grads = []
  noisy_grads = []
  clean_outputs = []
  noisy_outputs = []
  for i in range(epochs//frequency):
    print(f"llr accuracy:{model.last_layer_reweight(random_loader,score=True)}")
    clean_grads.append(torch.zeros(len(clean_set),32)) # initialize tensor for grads for whole dataset at this epoch
    clean_outputs.append(torch.zeros(len(clean_set),32))
    for j,(x,y) in enumerate(clean_set):
      x = torch.unsqueeze(x, 0).to(device)
      y = torch.unsqueeze(torch.tensor(y).to(device),0)
      optimizer.zero_grad()
      pred = model(x)
      loss = loss_f(pred,y)
      loss.backward()
      clean_grads[i][j,:] = grad.clone()
      clean_outputs[i][j,:] = output.clone().detach()

    noisy_grads.append(torch.zeros(len(noisy_set),32))
    noisy_outputs.append(torch.zeros(len(noisy_set),32))
    for j,(x,y) in enumerate(noisy_set):
      x = torch.unsqueeze(x, 0).to(device)
      y = torch.unsqueeze(torch.tensor(y).to(device),0)
      optimizer.zero_grad()
      pred = model(x)
      loss = loss_f(pred,y)
      loss.backward()
      noisy_grads[i][j,:] = grad.clone()
      noisy_outputs[i][j,:] = output.clone().detach()

    model.train(frequency,in_loader,optim=optimizer) # train model normally in in_set for f epochs
  
  i = i+1
  # get grads for final time
  clean_grads.append(torch.zeros(len(clean_set),32))
  clean_outputs.append(torch.zeros(len(clean_set),32))
  for j,(x,y) in enumerate(clean_set):
    x = torch.unsqueeze(x, 0).to(device)
    y = torch.unsqueeze(torch.tensor(y).to(device),0)
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_f(pred,y)
    loss.backward()
    clean_grads[i][j,:] = grad.clone()
    clean_outputs[i][j,:] = output.clone().detach()


  noisy_grads.append(torch.zeros(len(noisy_set),32))
  noisy_outputs.append(torch.zeros(len(noisy_set),32))
  for j,(x,y) in enumerate(noisy_set):
    x = torch.unsqueeze(x, 0).to(device)
    y = torch.unsqueeze(torch.tensor(y).to(device),0)
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_f(pred,y)
    loss.backward()
    noisy_grads[i][j,:] = grad.clone()
    noisy_outputs[i][j,:] = output.clone().detach()

  clf = model.last_layer_reweight(random_loader) # fit LLR for random set
  llr_weights = torch.squeeze(torch.tensor(clf.coef_)) # get weights
  
  # get inner product between weights 1 x w and all gradients n*epochs x w - > grad *  w.T = n x 1
  clean_dataset = pd.DataFrame({"point_id":np.arange(len(clean_grads[0]))}) # dataframe with points
  for i,grad in enumerate(clean_grads):
    prod = torch.matmul(grad.double(),llr_weights.T).cpu().numpy()
    clean_dataset[f"grad_{i}"] = prod # add tensor to pandas in format n x epochs (row - datapoint id, col - epoch)
    out = clean_outputs[i]
    prod = torch.matmul(out.double(),llr_weights.T).cpu().numpy()
    clean_dataset[f"out_{i}"] = prod
  
  noisy_dataset = pd.DataFrame({"point_id":np.arange(len(noisy_grads[0]))+len(clean_grads[0])})
  for i,grad in enumerate(noisy_grads):
    prod = torch.matmul(grad.double(),llr_weights.T).cpu().numpy()
    noisy_dataset[f"grad_{i}"] = prod
    out = noisy_outputs[i]
    prod = torch.matmul(out.double(),llr_weights.T).cpu().numpy()
    noisy_dataset[f"out_{i}"] = prod

  clean_dataset["type"] = "clean"
  noisy_dataset["type"] = "noisy"
  dataset = pd.concat([clean_dataset,noisy_dataset])
  
  # fit LLR for 100% correlation set
  # get weights
  # get inner product between weights and all gradients

  return dataset


class CNN(nn.Module):
  def __init__(self,temperature=1):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, 1)
    self.conv2 = nn.Conv2d(16, 32, 3, 1)
    self.conv3 = nn.Conv2d(32, 64, 3, 1)
    self.conv4 = nn.Conv2d(64, 32, 3, 1)
    
    self.pool =  nn.AdaptiveAvgPool2d(1)

    self.fc = nn.Linear(32, 2)

    self.temperature = temperature

    self.to(device)

  def get_embeds(self,x):
    x = self.conv1(x)
    x = F.relu(x)

    x = self.conv2(x)
    x = F.relu(x)

    x = self.conv3(x)
    x = F.relu(x)

    x = self.conv4(x)
    x = F.relu(x)
    x = self.pool(x)
    x = torch.flatten(x, 1)
    return x

  def forward(self,x):
    x = self.get_embeds(x)
    x = self.fc(x)
    x = x/self.temperature
    return x
  
  def train(self,epochs,loader,optim=None):
    if optim is None:
        optim = torch.optim.Adam(self.parameters())
    criterion = nn.CrossEntropyLoss()
    for i in range(epochs):
        total_loss=0
        for image, label in loader:
            image,label = image.to(device),label.to(device)
            optim.zero_grad()
            pred = self.forward(image)
            loss = criterion(pred,label)
            loss.backward()
            optim.step()
            total_loss+= loss.item()

  def train_batches(self,batches,loader,optim=None):
    if optim is None:
        optim = torch.optim.Adam(self.parameters())
    criterion = nn.CrossEntropyLoss()
    i = 0
    while i<batches:
      for image,label in loader:
        if i==batches:
          break
        image,label = image.to(device),label.to(device)
        optim.zero_grad()
        pred = self.forward(image)
        loss = criterion(pred,label)
        loss.backward()
        optim.step()
        i += 1
  
  def last_layer_reweight(self,loader,score=False):
    all_embeddings = []
    all_y = []
    with torch.no_grad():
        for image, label in loader:
            image,label = image.to(device),label.to(device)
            all_embeddings.append(self.get_embeds(image).detach().cpu().numpy())
            all_y.append(label.detach().cpu().numpy())
        all_embeddings = np.vstack(all_embeddings)
        all_y = np.concatenate(all_y)
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings,all_y,test_size=0.5)
    clf = LogisticRegression().fit(X_train, y_train)
    if not score:
      return clf
    return clf.score(X_test,y_test)

  def get_acc(self,loader):
    correct = 0
    length = 0
    with torch.no_grad():
        for image, label in loader:
            image,label = image.to(device),label.to(device)
            preds = self.forward(image)
            predictions = torch.argmax(preds,dim=1)
            correct += (predictions == label).float().sum().item()
            length += len(predictions)
        return correct/length