import torch
from torch import nn
from torchvision import transforms


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(1,1)
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x) # instead of Heaviside step fn
        return output
    
    
class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 8)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

        
        
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
#         self.batchnorm1 = nn.BatchNorm1d(512)
#         self.batchnorm2 = nn.BatchNorm1d(128)
#         self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
#         x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
#         x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
#         x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
    
    
        
class FeedforwardBin(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(FeedforwardBin, self).__init__()
        self.input_size = input_size
        self.hidden_size1  = hidden_size1
        self.hidden_size2  = hidden_size2
        self.hidden_size3 = hidden_size3
        
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size1)
        self.tahn1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        self.tahn2 = torch.nn.Tanh()            
        self.fc3 = torch.nn.Linear(self.hidden_size2, self.hidden_size3)
        self.tahn3 = torch.nn.Tanh()            
        self.fc4 = torch.nn.Linear(self.hidden_size3, 2)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_size1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)



        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)



    def forward(self, x):
        hidden1 = self.fc1(x)
        hidden1 = self.batchnorm1(hidden1)
        tahn1 = self.tahn1(hidden1)
        tahn1 = self.dropout1(tahn1)
        hidden2 = self.fc2(tahn1)     
        tahn2 = self.tahn2(hidden2)
        tahn2 = self.dropout2(tahn2)        
        fc3 = self.fc3(tahn2)
#         tahn3 = self.tahn3(fc3)
#         output = self.fc4(tahn3)

#         output = self.softmax(output)
        return fc3

