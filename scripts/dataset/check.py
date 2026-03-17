import torch
import os
import h5py

path = "/data2/mengzibing/medicine/datasets/dataset_offline/Sub-typing"
num_64=0
m,n=0,64
for filename in os.listdir(path):
    if filename.endswith(".h5"):
        with h5py.File(os.path.join(path, filename), 'r') as f:
            feature = f['features'][:]
            print(feature.shape)
            H,W,D=feature.shape
            if H>=64:
                num_64+=1
            m=max(m,H)
            n=min(n,H)
print(num_64,m,n)
