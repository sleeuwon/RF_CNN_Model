import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelsunshining import SParamCNNsunniest

def predicting(newinput):

    model = SParamCNNsunniest().to("cuda")
    model.load_state_dict(torch.load("state_dict_largerun.pt"))
    model.eval() #set the model to evaluation mode
    output = model(newinput.to("cuda"))
    return output
