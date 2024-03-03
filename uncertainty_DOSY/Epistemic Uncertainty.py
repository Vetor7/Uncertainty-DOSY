import random

import dataset
import numpy as np
import torch
from model import Transformer
import matplotlib.pyplot as plt
import torch.nn as nn

val_input = np.load("./Dataset/val_input.npy")
val_label = np.load("./Dataset/val_label.npy")
val_input = torch.from_numpy(val_input).float()
# val_label = torch.from_numpy(val_label).float()
sample = 0
cat_out = 32


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


with torch.no_grad():
    module = Transformer.load_from_checkpoint("Result/DOSY/last.ckpt")
    module.cpu()
    module.eval()
    module.freeze()


test_input = val_input[sample].unsqueeze(0)
test_input = test_input.to(torch.float32)

test_out = module(test_input)
test_out1 = test_out

module.apply(apply_dropout)

for i in range(50):
    test_out_x = module(test_input)
    test_out = torch.concat([test_out, test_out_x], 0)

var, test_out = torch.var_mean(test_out, 0)
test_out = test_out.unsqueeze(0)
var = var.unsqueeze(0)

var2 = var/test_out

test_out_addvar = test_out + var
test_out_subvar = test_out - var

test_out = test_out * test_input[:, :, 0][:, :, np.newaxis] / 3
test_out1 = test_out1 * test_input[:, :, 0][:, :, np.newaxis] / 3
test_out_addvar = test_out_addvar * test_input[:, :, 0][:, :, np.newaxis] / 3
test_out_subvar = test_out_subvar * test_input[:, :, 0][:, :, np.newaxis] / 3

test_out = test_out[0].cpu().detach().numpy()
test_out1 = test_out1[0].cpu().detach().numpy()
test_out_addvar = test_out_addvar[0].cpu().detach().numpy()
test_out_subvar = test_out_subvar[0].cpu().detach().numpy()

var = var[0].cpu().detach().numpy()
var2 = var2[0].cpu().detach().numpy()
test_out[test_out < np.tile(((np.max(test_out, axis=1)) * 0.7)[:, np.newaxis], [1, 140])] = 0
test_out1[test_out1 < np.tile(((np.max(test_out1, axis=1)) * 0.7)[:, np.newaxis], [1, 140])] = 0
test_out_addvar[test_out_addvar < np.tile(((np.max(test_out_addvar, axis=1)) * 0.7)[:, np.newaxis], [1, 140])] = 0
test_out_subvar[test_out_subvar < np.tile(((np.max(test_out_subvar, axis=1)) * 0.7)[:, np.newaxis], [1, 140])] = 0
var2[var2 < 10] = 0


fig1, axs = plt.subplots(3, 2, figsize=(9, 6))

axs[0][0].contour(val_label[sample].T)
axs[0][0].set_title('label')

axs[0][1].contour(test_out1.T)
axs[0][1].set_title('out')

axs[1][0].contour(var.T, cmap='viridis')
axs[1][0].set_title('out_var')

axs[1][1].contour(test_out.T)
axs[1][1].set_title('out_mean')

# plt.figure(5)
axs[2][0].contour(test_out_subvar.T, cmap=plt.cm.hot)
axs[2][0].set_title('sub_var')

axs[2][1].contour(test_out_addvar.T)
axs[2][1].set_title('add_var')

fig2, axs1 = plt.subplots(2, 2, figsize=(9, 6))

axs1[0][0].plot(val_label[sample][:, cat_out], color='black')
axs1[0][0].set_title('label')

axs1[0][1].plot(test_out1[:, cat_out - 2:cat_out + 3])
axs1[0][1].set_title('out')

axs1[1][0].plot(var[:, cat_out - 2:cat_out + 3])
axs1[1][0].set_title('out_var')

axs1[1][1].plot(test_out[:, cat_out - 2:cat_out + 3])
axs1[1][1].set_title('out_mean')

# plt.figure(5)
axs1[0][0].fill_between(np.linspace(0, 300, 300), np.max(test_out_subvar[:, cat_out - 3:cat_out + 3], -1),
                        np.max(test_out_addvar[:, cat_out - 3:cat_out + 3], -1), color='green', alpha=0.5)
# axs1[2][0].set_title('sub_var')


# axs1[0][0].plot(test_out_addvar[:,99].T,color='green')
# axs1[2][1].set_title('add_var')

plt.show()
