
"""model details of multi-layer perceptron"""

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

"""pytorch related"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class SignalClassificationBase(nn.Module):

    def model_init(self):
        """Initialize the model"""
        model = SpectralClassification()
        model = model.cuda()
        # summary(model, (1, 3648))
        return model

    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(self,data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)                # get the index of the max log-probability
        return torch.tensor(torch.sum(preds == labels).item() / len(preds)) , preds
    
    def training_step(self, batch):
        inputs, labels = batch
        inputs = inputs.float()
        labels = torch.flatten(labels)
        outputs = self(inputs)                              # generate predictions (forward pass)
        loss = F.cross_entropy(outputs, labels)             # calculate loss
        acc_train, preds = self.accuracy(outputs, labels)   # calculate accuracy
        return {'train_loss': loss, 'train_acc': acc_train}
    
    def train_epoch_end(self, outputs):
        """ Compute the train average loss and accuracy for each epoch
        """
        batch_losses = [x['train_loss'] for x in outputs]   # get all the batch loss
        epoch_loss = torch.stack(batch_losses).mean()       # combine losses
        batch_accs = [x['train_acc'] for x in outputs]      # get all the batch accuracy
        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracy
        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}
    
    def validation_step(self, batch):
        inputs, labels = batch
        inputs = inputs.float()
        labels = torch.flatten(labels)
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc_val, preds = self.accuracy(outputs, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc_val}
    
    def validation_epoch_end(self, outputs):
        """this is for computing the validation average loss and acc for each epoch"""
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch,train_result,val_result):
        """Print out the results after every epoch"""
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss : {:.4f}, val_acc: {:.4f}".format(
            epoch, train_result['train_loss'], train_result ['train_acc'], val_result['val_loss'], val_result['val_acc']
        ))

    def test_end(self, outputs):
        """For test set, it outputs average loss and accuracy, and output the predictions """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()                           # combined loss
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()                              # combined accuracy
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item() }  #, 'test_preds':batch_preds, 'test_labels':batch_labels }
    
    def test_predict(self, model, test_loader):
        """Loads the model and performs test evaluation"""
        with torch.no_grad():
            model.eval()
            outputs = [model.validation_step(batch) for batch in test_loader]       # perform testing for each batch
            results = model.test_end(outputs)
            print('test_loss:{:.4f}, test_acc: {:.4f}'.format(results['test_loss'], results['test_acc']))
        return results
    
        
    def evaluate(self, model, val_loader):
        with torch.no_grad():
            model.eval()
            outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)
    
    def fit(self, epochs, lr, model, train_loader, val_loader, opt_function = torch.optim.SGD):
        """Set fixed random seed"""
        torch.manual_seed(42)
        history = {}
        optimizer = opt_function(model.parameters(), lr)
        for epoch in range(epochs):
            
            # training Phase 
            model.train()
            train_outputs = []
            for batch in train_loader:
                optimizer.zero_grad()                   # reset the gradients
                outputs = model.training_step(batch)
                loss  = outputs['train_loss']
                train_outputs.append(outputs)
                loss.backward()
                optimizer.step()
            
            # TODO: update the comments 
            train_results = model.train_epoch_end(train_outputs)
            val_results = model.evaluate(model, val_loader)
            model.epoch_end(epoch, train_results, val_results)  

            to_add = {'train_loss': train_results['train_loss'], 'train_acc': train_results['train_acc'], 
                      'val_loss': val_results['val_loss'], 'val_acc': val_results['val_acc']}

            for key, val in to_add.items():
                if key in history :
                    history[key].append(val)
                else:
                    history[key] = [val]
        return history
    
    def plot_acc_loss(self,history, test_acc,test_loss,lr ):
        """Plot accuracy and loss in the same figure"""

        path = "results"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        new_path = "./" + path + "\\"
       
        test_accmean  = np.mean(test_acc)
        test_accstd = np.std(test_acc)

        test_lossmean = np.mean(test_loss)
        test_lossstd = np.std(test_loss)

        f, (ax1, ax2) = plt.subplots(2,1,figsize = (5,10))
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        # test_acc = test_accuracy["test_acc"]
        ax1.plot(train_acc, label = 'Train Accuracy', c ="#0B032D")
        ax1.plot(val_acc, label = 'Validation Accuracy' , c = "#F67E7D")

        ax1.errorbar(0,test_accmean, yerr=test_accstd, fmt="-o", label = 'Test Accuracy' , c= "#843B62")
        ax1.set_ylabel('Accuracy Value')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Accuracy')
        ax1.set_ylim(0,1.1)
        legend_acc = ax1.legend(loc = "lower right")

        train_loss = history['train_loss']
        val_loss = history['val_loss']
        test_loss = test_loss
        # test_loss = test_accuracy["test_loss"]
        ax2.plot(train_loss, label = 'Train Loss', c ="#0B032D")
        ax2.plot(val_loss, label = 'Validation Loss', c = "#F67E7D")
        ax2.errorbar(0,test_lossmean ,yerr=test_lossstd,fmt= "-o" , label = 'Test Loss', c= "#843B62")
        ax2.set_ylabel('Loss Value')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss')
        legend_loss = ax2.legend(loc = "best")

        plt.savefig(new_path+ "result_" + str(lr)+ ".png",bbox_inches='tight')

class SpectralClassification(SignalClassificationBase): 

    def __init__(self,loss_mode = "cross_entropy"):

        super().__init__()

        self.input_channel = 1301               
        self.out_channel = 2                    
        self.loss_mode = loss_mode

        self.fc1 = nn.Linear(self.input_channel, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, self.out_channel)
        self.relu  = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        
        if self.loss_mode == "NLLLoss":
            # adding softmax layer
            x = self.softmax(x)
        elif self.loss_mode == "cross_entropy":
            x = x        
        return x
