import torch
data = torch.load('/root/autodl-tmp/nepa/datas/TCGA/TCGA-BLCA-PT/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.pt')
print(list(data.keys()))
for key in data.keys():
    print(key,':',data[key])
    