import h5py

h5_path = "/data2/mengzibing/medicine/datasets/dataset_o/Sub-typing/BRCA/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.h5"



with h5py.File(h5_path, 'r') as f:
    print("Keys in the H5 file:", list(f.keys()))
    for key in f.keys():
        print(f"Shape of '{key}': {f[key].shape}")
