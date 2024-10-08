import torch
import pickle
from torch.utils.data import TensorDataset
import numpy as np

DATASET_DIR = './'
DESIGN_SIZE = 40

def load_dataset():
    """
    Loads the dataset from pickle files, converts it to torch Tensors,
    and returns a TensorDataset.
    """
    # Load input pixel array
    with open(DATASET_DIR + 'input_pixel_array.pkl', 'rb') as f:
        pixel_array = pickle.load(f)

    # Load output s-parameter array
    with open(DATASET_DIR + 'output_sparam_array.pkl', 'rb') as f:
        sparam_array = pickle.load(f)

    # Ensure both arrays have the same number of samples
    assert len(pixel_array) == len(sparam_array), "Number of samples mismatch between input and output arrays"

    # Combine real and imaginary parts along a new dimension
    sparam_array_combined = np.concatenate((sparam_array.real[..., np.newaxis], sparam_array.imag[..., np.newaxis]), axis=-1)

    inputs = torch.tensor(pixel_array[:-10], dtype=torch.float32)
    outputs = torch.tensor(sparam_array_combined[:-10], dtype=torch.float32)

    dataset = TensorDataset(inputs, outputs)

    return dataset
