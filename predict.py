import torch
from model import SParamCNN

def predict(new_input, name):
    model = SParamCNN()
    model.load_state_dict(torch.load(name))
    model.eval() #set the model to evaluation mode
    with torch.no_grad(): #we do not need to track gradients
        output1,output2 = model(new_input)
    return output1,output2
