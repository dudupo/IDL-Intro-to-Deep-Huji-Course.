import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np





def iterate_datasets( ):
    datafiles = [   'A0101_pos.txt',
                    'A0201_pos.txt',
                    'A0203_pos.txt',
                    'A0207_pos.txt',
                    'A0301_pos.txt',
                    'A2402_pos.txt',
                    'negs.txt' ]

    for datafile in datafiles: 
        with open( f'../data/ex1 data/{datafile}') as _file:
            pos =  1 if "pos" in  datafile else 0
            for line in _file.readlines():
                yield line.strip(), pos


def Count_Terminals_in_Data():

    datafiles = [   'A0101_pos.txt',
                    'A0201_pos.txt',
                    'A0203_pos.txt',
                    'A0207_pos.txt',
                    'A0301_pos.txt',
                    'A2402_pos.txt',
                    'negs.txt' ]
    
    _terminals = set()
    for datafile in datafiles: 
        with open( f'../data/ex1 data/{datafile}') as _file:
            for terminal in _file.read():
                _terminals.add(terminal)
    if '\n' in _terminals: 
        _terminals.remove( '\n' )
    return _terminals


def indexsize( _set ):
    return { val:key for key,val in enumerate(_set) }

def pharse_line(line, _terminals):
    _map = indexsize(_terminals)
    encoding =  [ float(0) for _ in range(len(line) * len(_map))] 
    for i, char in enumerate(line): 
        encoding[ i*len(_map) + _map[char] ] = float(1)
        
    return encoding


def load_datasets():
    X, y = [], []
    _terminals = Count_Terminals_in_Data()
    for encoding, pos in iterate_datasets():
        X.append( pharse_line(encoding, _terminals) )
        y.append( [float(pos)] )
    return (torch.from_numpy(np.array(_)) for _ in [X,y])

def pharse_line_test( ):
    _terminals = Count_Terminals_in_Data()
    #print(pharse_line("QLDAELLRY", _terminals))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9*20, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()

        )
        self.double()

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def train(data, model, loss_fn, optimizer):
    X, y = data 
    
    model.train() #set training mod 
    # Compute prediction error

    print(X.shape)
    print(y.shape)
    pred = model(X)
    print(pred.shape)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #if batch % 100 == 0:
    loss  = loss.item()
    print(f"loss: {loss:>7f}") # [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    _termianls  = Count_Terminals_in_Data() 
    pharse_line_test()
    X, y = load_datasets()

    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train((X,y), model, loss_fn, optimizer)


