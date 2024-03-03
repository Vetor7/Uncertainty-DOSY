import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as scio
from scipy import interpolate

val_input = np.load("./Dataset/val_input.npy")
val_label = np.load("./Dataset/val_label.npy")
NmrData = scio.loadmat("./Dataset/" + "M6" + "_net_input.mat")


sample = np.random.randint(100)
test_input = val_input[sample, :, :]
# test_input = torch.tensor(test_input)
# test_input = test_input.to(torch.float32)



# NmrDatai = np.zeros([NmrData['S'].shape[0], 30])
# for i in np.arange(NmrData['S'].shape[0]):
#     f = interpolate.interp1d(NmrData['b'][0], NmrData['S'][i, :], fill_value='extrapolate')
#     NmrDatai[i] = f(np.linspace(0, np.max(NmrData['b'][0]), 30))
# test_input = NmrDatai
test_input = test_input/test_input[ :, 0][:,np.newaxis]

plt.plot(np.linspace(0, 3, 30), test_input.T)
print(np.min(test_input))
plt.figure(1)

#
# X, Y = np.meshgrid(np.linspace(0, 14, 140), np.linspace(0, 3, 300))
# plt.contour(Y, X, val_label[sample], [0.1, 0.2, 0.4, 0.7, 1], colors='k')

plt.xlabel('Chemical Shift(ppm)')
plt.ylabel('diffusion coefficients')
plt.title('Ideal DOSY')

plt.show()
