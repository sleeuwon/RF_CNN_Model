import numpy as np
import pandas as pd
from model import SParamCNN
from predict import predict
import pickle
import torch
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error


import matplotlib.pyplot as plt

DATASET_DIR = './'
DESIGN_SIZE = 40

with open(DATASET_DIR + 'input_pixel_array.pkl', 'rb') as f:
    pixel_array = pickle.load(f)
pixel_array.shape  # [num_of_designs, num_of_layers(3), 40, 40]
with open(DATASET_DIR + 'output_sparam_array.pkl', 'rb') as f:
    sparam_array = pickle.load(f)
sparam_array.shape #[num_of_designs, num_of_sparameters(12), frequency_points(0GHz-300GHz)]

FREQ_RANGE = list(range(0, sparam_array.shape[2]))
FREQ_RANGE[0], FREQ_RANGE[-1]

#################################################################################################
# Technically we should have s11, s12, s13, s14, s21, s22, s23, s24, s31, s32, s33, s34, s41, s42, s43, s44. All of them has real/imaginary part.
# But because our design is symmetrical, we actually only have 12 different unique s-parameters !!!
# s11 == s22
# s12 == s21
# s13 == s31 == s24 == s42
# s14 == s41 == s23 == s32
# s33 == s44
# s34 == s43

SPARAM_LIST_NAME = ['s11_real', 's11_imag', 's12_real', 's12_imag', 's13_real', 's13_imag', 's14_real', 's14_imag', 
  's33_real', 's33_imag', 's34_real', 's34_imag']

sample_index = 0
my_sample_pixel_array = pixel_array[sample_index,:,:,:]

my_sample_sparam_array = sparam_array[sample_index,:,:]
inputs = torch.tensor(sparam_array, dtype=torch.float32)
print(len(inputs))
#r2 = [[r2_score(sparam_array[i,j,:], outputs[i,j,:,0]) for i in range(len(outputs))] for j in range(len(outputs[0]))]
#print(len(r2))
#print(len(r2[0]))
#r2_avg = [np.mean(r2[i][-10:]) for i in range(len(r2))]
#print(r2_avg)
#MAPE calc
output1,output2 = (predict(inputs,'statedict_reverselargepure.pt'))
output1=output1.to("cpu").numpy()
output2=output2.to("cpu").numpy()
print(len(output1))
print(len(output1[0]))
print(len(output1[0][0]))
print(len(output1[0][0][0]))
print(len(output2))
print(len(output2[0]))
print(len(output2[0][0]))
print(len(output2[0][0][0]))
print(len(sparam_array))
print(len(sparam_array[0]))
print(len(sparam_array[0][0]))

#mapemae = [[mean_absolute_percentage_error(sparam_array[-10:,j,i], output1[-10:,j,i,0]) for i in range(len(output1[0][0]))] for j in range(len(output1[0]))]
#maemae = [[mean_absolute_error(sparam_array[:,j,i], output1[:,j,i,0]) for i in range(len(output1[0][0]))] for j in range(len(output1[0]))]
#rmsemae = [[root_mean_squared_error(sparam_array[:,j,i], output1[:,j,i,0]) for i in range(len(output1[0][0]))] for j in range(len(output1[0]))]



my_sample_pixel_array.shape, my_sample_sparam_array.shape
output2 = output2[sample_index]

cols, rows = 3, 2
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

axes[0,0].set_title('ground layer, sample--'+ str(sample_index))
axes[0,0].imshow(my_sample_pixel_array[0,:,:], cmap='Greys')
axes[0,1].set_title('bottom layer, sample--'+ str(sample_index))
axes[0,1].imshow(my_sample_pixel_array[1,:,:], cmap='Greys')
axes[0,2].set_title('top layer, sample--'+ str(sample_index))
axes[0,2].imshow(my_sample_pixel_array[2,:,:], cmap='Greys')

axes[1,0].imshow(np.around(output2[0,:,:],decimals=0), cmap='Blues')
axes[1,1].imshow(np.around(output2[1,:,:],decimals=0), cmap='Blues')
axes[1,2].imshow(np.around(output2[2,:,:],decimals=0), cmap='Blues')

plt.tight_layout()
plt.show()





output1 = output1[sample_index]
rows, cols = 6, 2
fig, axes = plt.subplots(rows, cols, figsize=(cols*6,rows*3))

for i in range(rows):
    s_real_name = SPARAM_LIST_NAME[2*i]
    axes[i, 0].set_title(s_real_name+' over frequency, sample--'+ str(sample_index))
    axes[i, 0].set_xlabel('frequency (GHz)')
    axes[i, 0].plot(FREQ_RANGE, my_sample_sparam_array[2*i, :], color='r')
    axes[i, 0].plot(FREQ_RANGE, output1[2*i, :, 0], color='g')
    axes[i, 0].set_ylim(-1,1)
    

    s_imag_name = SPARAM_LIST_NAME[2*i+1]
    axes[i, 1].set_title(s_imag_name+' over frequency, sample--'+ str(sample_index))
    axes[i, 1].set_xlabel('frequency (GHz)')
    axes[i, 1].plot(FREQ_RANGE, my_sample_sparam_array[2*i+1,:],color='r')
    axes[i, 1].plot(FREQ_RANGE, output1[2*i+1, :, 0], color='g')
    axes[i, 1].set_ylim(-1,1)



plt.tight_layout()
plt.show()
print(len(rmsemae))
print(len(rmsemae[0]))
"""
fig, axes = plt.subplots(rows, cols, figsize=(cols*6,rows*3))

for i in range(12):
    s_name = SPARAM_LIST_NAME[i]
    axes[int(i/2), int(i%2)].set_title(s_name+' MAPE over frequency, error')
    axes[int(i/2), int(i%2)].set_xlabel('frequency (GHz)')
    axes[int(i/2), int(i%2)].plot(FREQ_RANGE, mapemae[i][:], color='g')
    axes[int(i/2), int(i%2)].set_ylim(0,1)
plt.tight_layout()

plt.show()
"""
fig, axes = plt.subplots(rows, cols, figsize=(cols*6,rows*3))

for i in range(12):
    s_name = SPARAM_LIST_NAME[i]
    axes[int(i/2), int(i%2)].set_title(s_name+' RMSE over frequency, error')
    axes[int(i/2), int(i%2)].set_xlabel('frequency (GHz)')
    axes[int(i/2), int(i%2)].plot(FREQ_RANGE, rmsemae[i][:], color='g')
    axes[int(i/2), int(i%2)].set_ylim(0,0.1)
plt.tight_layout()

plt.show()
fig, axes = plt.subplots(rows, cols, figsize=(cols*6,rows*3))

for i in range(12):
    s_name = SPARAM_LIST_NAME[i]
    axes[int(i/2), int(i%2)].set_title(s_name+' MAE over frequency, error')
    axes[int(i/2), int(i%2)].set_xlabel('frequency (GHz)')
    axes[int(i/2), int(i%2)].plot(FREQ_RANGE, maemae[i][:], color='g')
    axes[int(i/2), int(i%2)].set_ylim(0,0.1)
plt.tight_layout()

plt.show()


"""
plt.bar(SPARAM_LIST_NAME, r2_avg, color='pink')
plt.xlabel('S params')
plt.ylabel('r^2 average ')
plt.title('over 10 sample testing data')
plt.show()



"""
