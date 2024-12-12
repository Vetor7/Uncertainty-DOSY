import os

# model
Type = os.getenv('TYPE', 'T1T2') 
n_layers = 10
n_heads = 7
lr = 1e-3
dropout = 0.1
label_size = 140
lr_scheduler='ReduceLR'

# main
gpu = '0'
batch_size = 48
mul_label = 3
max_epochs = 61

# dataset
save_Dataset = True
n_fre = 300
o_fre = 20
max_fre = 10
num_D = 3
dB = 30
min_sep = 0.5
num_samples = 30000
ratio = 0.1
sig = 0.08
base_D = 0
sig_lorz=0.06
max_b = 0.8
max_D = 14
signal_dim = 30
# T1T2
if Type == 'T1T2':
    n_fre = 1
    num_D = 1
    max_b = 20
    max_D = num_D
    signal_dim = 10
    min_sep =  0.1
    dB = num_D * 30
    sig = 0.04
    n_layers = 5
    num_samples = 200000
    batch_size = 512
    max_epochs = 40