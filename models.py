import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

class model():
    # TODO:test
    def last_layer_reweight(self):
        for param in self.NN.parameters():
            param.requires_grad = False
        self.NN.fc.reset_parameters()
        self.NN.fc.requires_grad = True
    def reset(self):
        for param in self.NN.parameters():
            param.reset_parameters()    
    def train(self,epochs,dataset,verbose):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.NN.parameters())
        losses = []
        accs = []
        for i in range(epochs):
            x,y=dataset[:]
            optimizer.zero_grad()

            preds = self.NN(x.float())
            loss = criterion(preds,y.long())
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            predictions = torch.argmax(preds,dim=1)
            correct = (predictions == y).float().sum()
            accs.append(correct/len(y))
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(losses)
            fig.show()
            ax.plot(accs)
            fig.show()

class bottleNN(model):
    def __init__(self,hidden_units,bottleneck,out=2):
        self.bottleneck = bottleneck
        self.out = out
        self.hidden_units = hidden_units
        self.NN = torch.nn.Sequential(OrderedDict([
          ('embeds',torch.nn.Sequential(torch.nn.Linear(3, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units,bottleneck),
            torch.nn.ReLU())),
          ('fc',torch.nn.Linear(bottleneck, out))
          ])  
      )
    
    



def get_acc(NN,dataset):
  with torch.no_grad():
    x,y = dataset[:]
    preds = NN(x.float())
    predictions = torch.argmax(preds,dim=1)
    correct = (predictions == y).float().sum()
    return correct/len(y)
