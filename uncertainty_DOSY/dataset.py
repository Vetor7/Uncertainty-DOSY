import os
import torch
import numpy as np
from scipy import interpolate

from tqdm import trange
from sklearn.model_selection import train_test_split as train_val

import config

np.random.seed(42)


def load_dataloader(batch_size, num_D):
    print("Begin to generate simulation signals")

    DOSY_signal = gen_signal_dosy(num_D=num_D)

    clean_signals, label = DOSY_signal.gen_signal_2D()

    train_input, val_input, train_label, val_label = train_val(clean_signals, label,
                                                               test_size=config.ratio, random_state=42)

    train_input = torch.from_numpy(train_input).float()
    val_input = torch.from_numpy(val_input).float()
    train_label = torch.from_numpy(train_label).float()
    val_label = torch.from_numpy(val_label).float()
    # train_DF = torch.from_numpy(train_DF).float()
    # val_DF = torch.from_numpy(val_DF).float()
    # train_k = torch.from_numpy(train_k).float()
    # val_k = torch.from_numpy(val_k).float()

    if config.save_Dataset == True:
        output_dir_dataset = "./Dataset/"
        np.save(os.path.join(output_dir_dataset, "train_input"), train_input)
        np.save(os.path.join(output_dir_dataset, "val_input"), val_input)
        np.save(os.path.join(output_dir_dataset, "train_label"), train_label)
        np.save(os.path.join(output_dir_dataset, "val_label"), val_label)
        # np.save(os.path.join(output_dir_dataset, "train_DF"), train_DF)
        # np.save(os.path.join(output_dir_dataset, "val_DF"), val_DF)
        # np.save(os.path.join(output_dir_dataset, "train_k"), train_k)
        # np.save(os.path.join(output_dir_dataset, "val_k"), val_k)

    train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                               persistent_workers=False, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_input, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0,
                                             persistent_workers=False, shuffle=False)

    print('successfully load dataloader')

    return train_loader, val_loader


def load_dataloader_exist(batch_size):
    print("Begin to generate simulation signals")

    train_input = np.load("./Dataset/train_input.npy")
    train_label = np.load("./Dataset/train_label.npy")
    val_input = np.load("./Dataset/val_input.npy")
    val_label = np.load("./Dataset/val_label.npy")
    # train_k = np.load("./Dataset/train_k.npy")
    # train_DF = np.load("./Dataset/train_DF.npy")
    # val_k = np.load("./Dataset/val_k.npy")
    # val_DF = np.load("./Dataset/val_DF.npy")

    train_input = torch.from_numpy(train_input).float()
    val_input = torch.from_numpy(val_input).float()
    train_label = torch.from_numpy(train_label).float()
    val_label = torch.from_numpy(val_label).float()
    # train_DF = torch.from_numpy(train_DF).float()
    # train_k = torch.from_numpy(train_k).float()
    # val_k = torch.from_numpy(val_k).float()
    # val_DF = torch.from_numpy(val_DF).float()

    train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                               persistent_workers=False, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_input, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0,
                                             persistent_workers=False, shuffle=False)

    print('successfully load dataloader')

    return train_loader, val_loader


class gen_signal_dosy():
    def __init__(self, max_D=config.max_D, label_size=config.label_size, min_sep=config.min_sep, num_D=config.num_D
                 , max_b=config.max_b, signal_dim=config.signal_dim, dB=config.dB, sig_lorz=config.sig_lorz,
                 max_fre=config.max_fre, n_fre=config.n_fre, num_samples=config.num_samples, sig=config.sig):
        self.max_D = max_D
        self.label_size = label_size
        self.min_sep = min_sep
        self.num_D = num_D
        self.max_b = max_b
        self.signal_dim = signal_dim
        self.dB = dB
        self.sig_lorz = sig_lorz
        self.max_fre = max_fre
        self.n_fre = n_fre
        self.num_samples = num_samples
        self.sig = sig
        self.set_dim = 30

    def gen_signal_2D(self):
        S = np.zeros([self.num_samples, self.set_dim, self.n_fre])
        label = np.zeros([self.num_samples, self.label_size, self.n_fre])
        DF = np.zeros([self.num_samples, self.label_size, self.num_D])
        Ci_f = np.zeros([self.num_samples, self.num_D, self.n_fre])
        S_kernel = np.zeros([self.num_samples, self.signal_dim, self.label_size])
        for i in trange(self.num_samples):
            Ci_f[i] = self.get_Ci_f()
            D = self.get_D()
            DF[i] = self.get_DF(D, self.sig)
            label[i] = self.get_label(DF[i], Ci_f[i])
            label[i] = label[i] / (np.sum(label[i], -2)[np.newaxis, :])
            S_kernel[i] = self.get_Skernel()
            S[i] = self.get_signal(S_kernel[i], label[i])

        S = S.swapaxes(1, 2)
        label = label.swapaxes(1, 2)
        noise_S = self.get_noise_S(S)
        # new_noise_S, new_S, S_kernel = self.data_cut(noise_S, S)

        return noise_S.astype('float32'), label.astype('float32')

    def get_DF(self, D, Sigma):  # D(3,1)
        D = D.reshape(self.num_D, 1)
        d = np.linspace(0, self.max_D, self.label_size)
        sqrt_2pi = np.power(2 * np.pi, 0.5)
        coef = 1 / (sqrt_2pi * Sigma)
        powercoef = -1 / (2 * np.power(Sigma, 2))
        mypow = powercoef * (np.power((d - D), 2))
        DF = coef * (np.exp(mypow))
        DF = DF / np.tile(np.max(DF, axis=1)[:, np.newaxis], [1, self.label_size])

        return DF.T

    def get_D(self):
        while True:  # TODO
            D_value = (np.random.random(self.num_D) * self.max_D) + config.base_D
            D_value = np.sort(D_value)
            D_value_t = np.roll(D_value, 1)
            if np.min(np.abs(D_value - D_value_t)) > self.min_sep:
                break

        return D_value

    def get_label(self, DF, Ci_f):
        label = np.matmul(DF, Ci_f)
        return label

    def get_Skernel(self):
        D_lab = np.linspace(0, self.max_D, self.label_size)

        max_b = self.max_b

        b = np.linspace(0, max_b, self.set_dim)
        S_kernel = np.exp(-b.reshape(self.set_dim, 1) * D_lab)

        return S_kernel

    def get_signal(self, S_k, label):
        signal = np.matmul(S_k, label)
        return signal

    def get_noise_S(self, S):
        snr = np.exp(np.log(10) * float(self.dB) / 10)
        num_samples, n_fre, signal_dim = np.shape(S)
        noise_S = np.zeros([num_samples, n_fre, signal_dim])
        sigma = np.sqrt(1. / snr)

        for i in trange(num_samples):
            noise = np.random.randn(n_fre, signal_dim)
            mult = sigma * np.linalg.norm(S[i, :, :], 2) / (np.linalg.norm(noise, 2))
            noise = noise * mult
            noise_S[i, :, :] = S[i, :, :] + noise
        return noise_S

    def get_Ci_f(self):
        Ci_f = np.zeros([self.num_D, self.n_fre])
        num_fre = np.zeros([self.num_D, 1])
        while (np.sum(num_fre, 0) < 18 or np.sum(num_fre, 0) > 22):
            num_fre = np.random.randint(0, 10, (self.num_D, 1))

        for i in np.arange(self.num_D):
            fre = np.tile(np.random.random([int(num_fre[i, 0]), 1]) * self.max_fre, [1, self.n_fre])
            chemical_shift = np.tile(np.linspace(0, self.max_fre, self.n_fre), [int(num_fre[i, 0]), 1])
            lorz = self.sig_lorz ** 2 / ((chemical_shift - fre) ** 2 + self.sig_lorz ** 2).reshape(int(num_fre[i, 0]),
                                                                                                   self.n_fre)
            lorz = lorz / (np.max(lorz, axis=1).reshape([int(num_fre[i, 0]), 1]))
            Ci_f[i] = np.sum(lorz, 0)
        return Ci_f

    def data_cut(self, noise_S, pure_s):
        b = np.linspace(0, self.max_b, self.set_dim)
        new_b = np.zeros(self.num_samples)
        new_bx = np.zeros([self.signal_dim])
        S_kernel = np.zeros([self.num_samples, self.signal_dim, self.label_size])
        D_lab = np.linspace(0, self.max_D, self.label_size)
        new_noise_S = np.zeros([self.num_samples, self.n_fre, self.signal_dim])
        new_pure_s = np.zeros([self.num_samples, self.n_fre, self.signal_dim])

        for i in trange(self.num_samples):
            noise_S[i, :, :] = noise_S[i, :, :] / (noise_S[i, :, 0][:, np.newaxis])  # 归一化
            for b_idx in np.arange(self.set_dim):
                if b_idx == self.set_dim - 1:
                    new_bx = np.linspace(0, self.max_b, self.signal_dim)
                    F = interpolate.interp1d(b, noise_S[i], fill_value='extrapolate')
                    f = interpolate.interp1d(b, pure_s[i], fill_value='extrapolate')
                    new_noise_S[i, :self.n_fre, :] = F(new_bx)
                    new_pure_s[i, :self.n_fre, :] = f(new_bx)

                elif np.min(noise_S[i, :, b_idx]) < 0.05:
                    Scat = noise_S[i, :, 0:b_idx]  # 截断
                    scat = pure_s[i, :, 0:b_idx]
                    new_b[i] = b[b_idx - 1]
                    new_bx = np.linspace(0, new_b[i], self.signal_dim)
                    F = interpolate.interp1d(b[0:b_idx], Scat[:, :], fill_value='extrapolate')
                    f = interpolate.interp1d(b[0:b_idx], scat[:, :], fill_value='extrapolate')
                    new_noise_S[i, :self.n_fre, :] = F(new_bx)
                    new_pure_s[i, :self.n_fre, :] = f(new_bx)
                    break

            S_kernel[i, :, :] = np.exp(-new_bx.reshape(self.signal_dim, 1) * D_lab)
        return new_noise_S, new_pure_s, S_kernel


if __name__ == '__main__':
    load_dataloader(config.batch_size, config.num_D)
