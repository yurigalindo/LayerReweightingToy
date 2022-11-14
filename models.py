import matplotlib.pyplot as plt
import torch
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
            self.contour_plot()
    def get_acc(self,dataset):
        with torch.no_grad():
            x,y = dataset[:]
            preds = self.NN(x.float())
            predictions = torch.argmax(preds,dim=1)
            correct = (predictions == y).float().sum()
            return correct/len(y)
    def contour_plot(self,points=50,min_range=-1,max_range=1):
        x = np.linspace(min_range,max_range, num=points,endpoint=True)
        y = np.linspace(min_range,max_range, num=points,endpoint=True)
        xx, yy = np.meshgrid(x, y)
        #z = z*np.ones(len(xx.flatten()))
        dataset = torch.t(torch.tensor(np.vstack([xx.flatten(),yy.flatten()])))
        out = self.NN(dataset.float())
        out = out.detach().numpy()
        out = out[:,0] - out[:,1]
        out = out.reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot()
        s = ax.contourf(x, y, out)
        ax.axis('scaled')
        #fig.set_size_inches(4, 4)
        fig.colorbar(s)
        fig.show()

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

class bottleNN_bias(bottleNN):
    def last_layer_reweight(self):
        super().last_layer_reweight()
        self.NN.fc.weight.requires_grad = False

class bottle_logistic(bottleNN):
    def __init__(self,hidden_units,bottleneck,in_dim=3,out=2, **logistic_args):
        super().__init__(hidden_units,bottleneck,in_dim,out)
        self.logistic = None
        self.logistic_args = logistic_args
    def last_layer_reweight(self):
        for param in self.NN.embeds.parameters():
            param.requires_grad = False
        self.logistic = LogisticRegression(**self.logistic_args)
        self.scaler = StandardScaler()
        
    def train(self,epochs,dataset,verbose):
        if self.logistic is None:
            super().train(epochs,dataset,verbose)
        else:
            # put a logistic on top
            x,y = dataset[:]
            with torch.no_grad():
                x = self.NN.embeds(x).detach().numpy()
            x = self.scaler.fit_transform(x)
            y = y.numpy()
            self.logistic.fit(x,y)
    def get_acc(self,dataset):
        if self.logistic is None:
            return super().get_acc(dataset)
        else:
            x,y = dataset[:]
            with torch.no_grad():
                x = self.NN.embeds(x).detach().numpy()
            x = self.scaler.transform(x)
            preds = self.logistic.predict(x)
            correct = (preds == y.numpy()).sum()
            return correct/len(y)
    def contour_plot(self,points=50,min_range=-1,max_range=1):
        pass


class resnet(model):
    def __init__(self,model,out=2,batch_size=128):
        model.fc = torch.nn.Linear(model.fc.in_features,out)
        self.original = copy.deepcopy(model)
        self.NN = model
        self.batch_size = batch_size
    def train(self,epochs,dataset,verbose):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.NN.parameters())
        losses = []
        accs = []
        total = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        for _ in range(epochs):
            cum_loss = 0
            cum_correct = 0
            for x,y in dataloader:
                optimizer.zero_grad()

                preds = self.NN(x)
                loss = criterion(preds,y.long())
                loss.backward()
                optimizer.step()

                cum_loss += loss.item()
                predictions = torch.argmax(preds,dim=1)
                cum_correct += (predictions == y).float().sum().item()
            losses.append(cum_loss)
            accs.append(cum_correct/total)
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(losses)
            fig.show()
            ax.plot(accs)
            fig.show()
    def get_acc(self,dataset):
        with torch.no_grad():
            total = len(dataset)
            dataloader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size*2,shuffle=True)
            cum_correct = 0
            for x,y in dataloader:
                preds = self.NN(x)
                predictions = torch.argmax(preds,dim=1)
                cum_correct += (predictions == y).float().sum().item()
            return cum_correct/total
    def last_layer_reweight(self):
        for param in self.NN.parameters():
            param.requires_grad = False
        for param in self.NN.fc.parameters():
            param.requires_grad = True
    def reset(self):
        self.NN = self.original

class resnet_logistic(resnet):
    def __init__(self,model,out=2,batch_size=128,**logistic_args):
        super().__init__(model,out,batch_size)
        self.logistic = None
        self.logistic_args = logistic_args
    def last_layer_reweight(self):
        for param in self.NN.parameters():
            param.requires_grad = False
        self.logistic = LogisticRegression(**self.logistic_args)
        self.scaler = StandardScaler()
    def train(self,epochs,dataset,verbose):
        if self.logistic is None:
            super().train(epochs,dataset,verbose)
        else:
            # put a logistic on top
            dataloader = torch.utils.data.DataLoader(dataset,batch_size=len(dataset)+1)
            x,y = next(iter(dataloader))
            with torch.no_grad():
                # get embeds
                x = self.NN.conv1(x)
                x = self.NN.bn1(x)
                x = self.NN.relu(x)
                x = self.NN.maxpool(x)

                x = self.NN.layer1(x)
                x = self.NN.layer2(x)
                x = self.NN.layer3(x)
                x = self.NN.layer4(x)

                x = self.NN.avgpool(x)
                x = torch.flatten(x, 1)
            x = x.detach().numpy()
            x = self.scaler.fit_transform(x)
            y = y.numpy()
            self.logistic.fit(x,y)
    def get_acc(self,dataset):
        if self.logistic is None:
            return super().get_acc(dataset)
        else:
            dataloader = torch.utils.data.DataLoader(dataset,batch_size=len(dataset)+1)
            x,y = next(iter(dataloader))
            with torch.no_grad():
                # get embeds
                x = self.NN.conv1(x)
                x = self.NN.bn1(x)
                x = self.NN.relu(x)
                x = self.NN.maxpool(x)

                x = self.NN.layer1(x)
                x = self.NN.layer2(x)
                x = self.NN.layer3(x)
                x = self.NN.layer4(x)

                x = self.NN.avgpool(x)
                x = torch.flatten(x, 1)
            x = x.detach().numpy()
            x = self.scaler.transform(x)
            preds = self.logistic.predict(x)
            correct = (preds == y.numpy()).sum()
            return correct/len(y)