import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset

DATASET_DIR = './'
DESIGN_SIZE = 40

def main():
    #load input pixel array
    with open(DATASET_DIR + 'input_pixel_array.pkl', 'rb') as f:
        pixel_array = pickle.load(f)

    #load output s-parameter array
    with open(DATASET_DIR + 'output_sparam_array.pkl', 'rb') as f:
        sparam_array = pickle.load(f)

    #processing to get complex tensor
    rows, cols = 6, 2
    for i in range(rows):
        real_sparam = sparam_array[:,2*i,:]
        imag_sparam = sparam_array[:,2*i+1,:]

    #convert numpy arrays to torch tensors
    inputs = torch.tensor(pixel_array, dtype = torch.float32)
    real = torch.tensor(real_sparam, dtype = torch.float32)
    imag = torch.tensor(imag_sparam, dtype = torch.float32)

    outputs = torch.complex(real,imag)
    #create a dataloader
    dataset = TensorDataset(inputs, outputs)
    print(torch.view_as_real(outputs))
