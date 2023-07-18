import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

class model():
    def last_layer_reweight(self):
        for param in self.NN.embeds.parameters():
            param.requires_grad = False
        
    def train(self,epochs,dataset,verbose=False,optim=None):
        criterion = torch.nn.CrossEntropyLoss()
        if optim is not None:
            optimizer = optim
        else:
            optimizer = torch.optim.Adam(self.NN.parameters())
        losses = []
        accs = []
        if epochs:
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
        else:
            last_loss = 0
            for i in range(100000):
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
                if abs(loss.item() - last_loss) <= 1e-6:
                    break
                last_loss = loss.item()

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
        plt.xlabel('Simple Feature')
        plt.ylabel('Complex Feature')
        plt.title('Contour Plot Before LLR')
        fig.colorbar(s)
        fig.show()

class bottleNN(model):
    def __init__(self,hidden_units,bottleneck,depth=1,in_dim=3,out=2):
        self.hidden_units = hidden_units
        self.bottleneck = bottleneck
        self.in_dim = in_dim
        self.out = out
        self.depth = depth
        self.reset()
    def reset(self):
        self.NN = torch.nn.Sequential(OrderedDict([
          ('embeds',torch.nn.Sequential(
                        torch.nn.Linear(self.in_dim, self.hidden_units),
                        torch.nn.ReLU(),
                        *[torch.nn.Sequential(
                            torch.nn.Linear(self.hidden_units,self.hidden_units),
                            torch.nn.ReLU()) 
                            for _ in range((self.depth-1))
                        ],
                        torch.nn.Sequential(
                            torch.nn.Linear(self.hidden_units,self.bottleneck),
                            torch.nn.ReLU()))
            ),
          ('fc',torch.nn.Linear(self.bottleneck, self.out))
          ])
        )

class bottleNN_bias(bottleNN):
    def last_layer_reweight(self):
        super().last_layer_reweight()
        self.NN.fc.weight.requires_grad = False

class bottle_logistic(bottleNN):
    def __init__(self,hidden_units,bottleneck,depth=1,in_dim=3,out=2, **logistic_args):
        super().__init__(hidden_units,bottleneck,depth,in_dim,out)
        self.logistic = None
        self.logistic_args = logistic_args
    def last_layer_reweight(self):
        self.logistic = LogisticRegression(**self.logistic_args)
        self.scaler = StandardScaler()
        
    def train(self,epochs,dataset,verbose=False,optim=None):
        if self.logistic is None:
            return super().train(epochs,dataset,verbose,optim=optim)
        # put a logistic on top
        x,y = dataset[:]
        with torch.no_grad():
            x = self.NN.embeds(x).detach().numpy()
        x = self.scaler.fit_transform(x)
        y = y.numpy()
        self.logistic.fit(x,y)
        if verbose:
            self.contour_plot()
    def get_acc(self,dataset):
        if self.logistic is None:
            return super().get_acc(dataset)
        x,y = dataset[:]
        with torch.no_grad():
            x = self.NN.embeds(x).detach().numpy()
        x = self.scaler.transform(x)
        preds = self.logistic.predict(x)
        correct = (preds == y.numpy()).sum()
        return correct/len(y)
    def contour_plot(self,points=50,min_range=-1,max_range=1):
        if self.logistic is None:
            return super().contour_plot(points,min_range,max_range)
        x = np.linspace(min_range,max_range, num=points,endpoint=True)
        y = np.linspace(min_range,max_range, num=points,endpoint=True)
        xx, yy = np.meshgrid(x, y)
        #z = z*np.ones(len(xx.flatten()))
        dataset = torch.t(torch.tensor(np.vstack([xx.flatten(),yy.flatten()])))
        with torch.no_grad():
            out = self.NN.embeds(dataset.float()).detach().numpy()
        out = self.scaler.transform(out)
        out = self.logistic.predict_proba(out)
        out = out[:,0] - out[:,1]
        out = out.reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot()
        s = ax.contourf(x, y, out)
        ax.axis('scaled')
        #fig.set_size_inches(4, 4)
        plt.xlabel('Simple Feature')
        plt.ylabel('Complex Feature')
        plt.title('Contour Plot After LLR')
        fig.colorbar(s)
        fig.show()


class resnet(model):
    def __init__(self,model,out=2,batch_size=128):
        model.fc = torch.nn.Linear(model.fc.in_features,out)
        self.original = copy.deepcopy(model)
        self.NN = model
        self.batch_size = batch_size
    def train(self,epochs,dataset,verbose=False):
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
    def train(self,epochs,dataset,verbose=False):
        if self.logistic is None:
            return super().train(epochs,dataset,verbose)
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


## MNIST

class CNN(nn.Module):
    def __init__(self,width=64,depth=3,bottleneck=32,in_dim=3,**logistic_args):
        super().__init__()
        self.in_dim = in_dim
        self.width = width
        self.depth = depth
        self.bottleneck = bottleneck
        self.logistic = None
        self.logistic_args = logistic_args
        self.reset()
    def reset(self):
        self.conv = nn.ModuleList()
        for i in range(self.depth):
            if i==0:
                self.conv.append(nn.Conv2d(self.in_dim,self.width,3,1))
            elif i==self.depth-1:
                # Last, bottleneck
                self.conv.append(nn.Conv2d(self.width,self.bottleneck,3,1))
            else:
                self.conv.append(nn.Conv2d(self.width,self.width,3,1,padding="same"))
        self.pool =  nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.bottleneck, 2)
        self.cuda()
    def get_embeds(self,x):
        for layer in self.conv:
            x = layer(x)
            x = F.relu(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return x
    def forward(self,x):
        x = self.get_embeds(x)
        x = self.fc(x)
        return x
    def train(self,epochs,train_loader,verbose=False,optim=None):
        if self.logistic is None:
            self._regular_train(epochs,train_loader,verbose,optim)
        else:
            self._llr_train(train_loader)
    def _regular_train(self,epochs,train_loader,verbose,optim):
        if optim is None:
            optim = torch.optim.Adam(self.parameters())
        past_loss = 0
        losses = []
        accs = []
        criterion = torch.nn.CrossEntropyLoss()
        if epochs:
            for i in range(epochs):
                total_loss=0
                correct=0
                length=0
                for image, label in train_loader:
                    image,label = image.cuda(),label.cuda()
                    optim.zero_grad()
                    pred = self(image)
                    loss = criterion(pred,label)
                    loss.backward()
                    optim.step()

                    total_loss+= loss.item()
                    predictions = torch.argmax(pred,dim=1)
                    correct += (predictions == label).float().sum().item()
                    length += len(label)
                if verbose:
                    losses.append(total_loss)
                    accs.append(correct/length)
        else:
            for i in range(1000):
                total_loss=0
                correct=0
                length=0
                for image, label in train_loader:
                    image,label = image.cuda(),label.cuda()
                    optim.zero_grad()
                    pred = self(image)
                    loss = criterion(pred,label)
                    loss.backward()
                    optim.step()

                    total_loss+= loss.item()
                    predictions = torch.argmax(pred,dim=1)
                    correct += (predictions == label).float().sum().item()
                    length += len(label)
                if verbose:
                    losses.append(total_loss)
                    accs.append(correct/length)
                if abs(total_loss-past_loss) < 1e-6:
                    break
                past_loss = total_loss
        if verbose:
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.plot(losses)
                fig.show()
                ax.plot(accs)
                fig.show()
    def _llr_train(self,loader):
        all_embeddings = []
        all_y = []
        for x, y in loader:
            with torch.no_grad():
                all_embeddings.append(self.get_embeds(x.cuda()).detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
        
        all_embeddings = np.vstack(all_embeddings)
        all_embeddings= self.scaler.fit_transform(all_embeddings)
        all_y = np.concatenate(all_y)
        self.logistic = self.logistic.fit(all_embeddings, all_y)
    def get_acc(self,loader):
        if self.logistic is None:
            with torch.no_grad():
                correct=0
                length=0
                for image, label in loader:
                    image,label = image.cuda(),label.cuda()
                    pred = self(image)
                    predictions = torch.argmax(pred,dim=1)
                    correct += (predictions == label).float().sum().item()
                    length += len(label)
                return correct/length
        else:
            all_embeddings = []
            all_y = []
            for x, y in loader:
                with torch.no_grad():
                    all_embeddings.append(self.get_embeds(x.cuda()).detach().cpu().numpy())
                    all_y.append(y.detach().cpu().numpy())
            
            all_embeddings = np.vstack(all_embeddings)
            all_embeddings= self.scaler.transform(all_embeddings)
            all_y = np.concatenate(all_y)
            return self.logistic.score(all_embeddings, all_y)
    def last_layer_reweight(self):
        self.logistic = LogisticRegression(**self.logistic_args)
        self.scaler = StandardScaler()
    