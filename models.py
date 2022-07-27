import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

class model():
    def last_layer_reweight(self):
        for param in self.NN.embeds.parameters():
            param.requires_grad = False 
    def train(self,epochs,dataset,verbose):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.NN.parameters())
        losses = []
        accs = []
        for _ in range(epochs):
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
    def get_acc(self,dataset):
        with torch.no_grad():
            x,y = dataset[:]
            preds = self.NN(x.float())
            predictions = torch.argmax(preds,dim=1)
            correct = (predictions == y).float().sum()
            return correct/len(y)

class bottleNN(model):
    def __init__(self,hidden_units,bottleneck,in_dim=3,out=2):
        self.hidden_units = hidden_units
        self.bottleneck = bottleneck
        self.in_dim = in_dim
        self.out = out
        self.reset()
    def reset(self):
        self.NN = torch.nn.Sequential(OrderedDict([
          ('embeds',torch.nn.Sequential(torch.nn.Linear(self.in_dim, self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units,self.bottleneck),
            torch.nn.ReLU())),
          ('fc',torch.nn.Linear(self.bottleneck, self.out))
          ])
        )
