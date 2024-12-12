import os
import torch
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split
import config as config

np.random.seed(42)  # Ensure reproducibility


def load_dataloader(batch_size):
    """Generates dataloaders for training and validation datasets."""
    print("Generating simulation signals")

    if config.Type == 'T1T2':
        clean_signals, labels = gen_signal_T1T2()
    else:
        DOSY_signal = gen_signal_dosy()
        clean_signals, labels, _, _ = DOSY_signal.gen_signal_2D()

    # Split data into training and validation sets
    train_input, val_input, train_label, val_label = train_test_split(
        clean_signals, labels, test_size=config.ratio, random_state=42)

    # Convert to PyTorch tensors
    train_input, val_input = map(lambda x: torch.from_numpy(x).float(), (train_input, val_input))
    train_label, val_label = map(lambda x: torch.from_numpy(x).float(), (train_label, val_label))

    # Save datasets to files if configured
    if config.save_Dataset:
        output_dir = "./Dataset/"
        os.makedirs(output_dir, exist_ok=True)
        for name, data in zip(["train_input", "val_input", "train_label", "val_label"],
                              [train_input, val_input, train_label, val_label]):
            np.save(os.path.join(output_dir, name), data.numpy())

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input, train_label),
        batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_input, val_label),
        batch_size=batch_size, shuffle=False)

    print('Dataloader loaded successfully.')
    return train_loader, val_loader


def load_dataloader_exist(batch_size):
    """Loads existing dataloaders from saved datasets."""
    print("Loading existing simulation signals")

    # Load datasets
    train_input, train_label, val_input, val_label = [
        torch.from_numpy(np.load(f"./Dataset/{name}.npy")).float()
        for name in ["train_input", "train_label", "val_input", "val_label"]
    ]

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input, train_label),
        batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_input, val_label),
        batch_size=batch_size, shuffle=False)

    print('Existing dataloader loaded successfully.')
    return train_loader, val_loader


class gen_signal_dosy():
    def __init__(self, **kwargs):
        # Initialize parameters with defaults from config or provided keyword arguments
        self.__dict__.update({k: getattr(config, k) for k in dir(config) if not k.startswith("__")})
        self.__dict__.update(kwargs)
    def update_parameters(self, **kwargs):
        # Update parameters with provided keyword arguments
        self.__dict__.update(kwargs)
    def gen_signal_2D(self):
        S = np.zeros([self.num_samples, self.signal_dim, self.n_fre])
        label = np.zeros([self.num_samples, self.label_size, self.n_fre])
        Ci_f = np.zeros([self.num_samples, self.num_D, self.n_fre])
        for i in trange(self.num_samples):
            Ci_f[i] = self.get_Ci_f()
            D = self.get_D()
            DF = self.get_DF(D, self.sig)
            label[i] = self.normalize_label(self.get_label(DF, Ci_f[i]))
            S_kernel = self.get_Skernel()
            S[i] = self.get_signal(S_kernel, label[i])

        noise_S = self.get_noise_S(S.swapaxes(1, 2))
        label = label.swapaxes(1, 2) * config.mul_label
        
        return noise_S.astype('float32'), label.astype('float32'), Ci_f.astype('float32'), S.swapaxes(1, 2).astype('float32')

    def normalize_label(self, label):
        """Normalize label intensities."""
        return label / np.sum(label, axis=-2, keepdims=True)

    def get_DF(self, D, Sigma):
        """Compute diffusion factors based on Gaussian distribution around D with spread Sigma."""
        d = np.linspace(0, self.max_D, self.label_size)
        D = D[:, np.newaxis]  # Ensure D is a column vector for broadcasting
        sqrt_2pi = np.sqrt(2 * np.pi)
        coef = 1 / (sqrt_2pi * Sigma)
        powercoef = -1 / (2 * Sigma ** 2)
        DF = coef * np.exp(powercoef * (d - D) ** 2)
        DF = DF / DF.max(axis=1, keepdims=True)

        return DF.T

    def get_D(self):
        """Generate unique diffusion coefficients with minimum separation."""
        while True:
            D = np.sort(np.random.rand(self.num_D) * self.max_D + config.base_D)
            if np.all(np.diff(D) > self.min_sep):
                return D

    def get_label(self, DF, Ci_f):
        """Calculate label intensities based on diffusion factors and concentration profiles."""
        return np.matmul(DF, Ci_f)

    def get_Skernel(self):
        """Generate signal kernel based on exponential decay model."""
        D_lab = np.linspace(0, self.max_D, self.label_size)
        b = np.linspace(0, self.max_b, self.signal_dim)
        return np.exp(-b[:, None] * D_lab)

    def get_signal(self, S_kernel, label):
        """Generate simulated signal by applying kernel to labels."""
        return np.matmul(S_kernel, label)

    # def get_noise_S(self, S):
    #     snr = np.exp(np.log(10) * float(self.dB) / 10)
    #     noise_S = np.zeros([self.num_samples, self.n_fre, self.signal_dim])
    #     sigma = np.sqrt(1. / snr)
    #
    #     for i in trange(self.num_samples):
    #         noise = np.random.randn(self.n_fre, self.signal_dim)
    #         mult = sigma * np.linalg.norm(S[i, :, :], 2) / (np.linalg.norm(noise, 2))
    #         noise = noise * mult
    #
    #         noise_S[i, :, :] = S[i, :, :] + noise
    #     return noise_S
    def get_noise_S(self, S):
        snr = np.exp(np.log(10) * float(self.dB) / 10)
        noise_S = np.zeros([self.num_samples, self.n_fre, self.signal_dim])
        sigma = np.sqrt(1. / snr)

        for i in trange(self.num_samples):
            for j in range(self.signal_dim):
                noise = np.random.randn(self.n_fre, 1)
                signal_energy_per_column = np.linalg.norm(S[i, :, j], 2)
                noise_energy_per_column = np.linalg.norm(noise, 2)
                mult = sigma * signal_energy_per_column / noise_energy_per_column
                noise *= mult
                noise_S[i, :, j] = S[i, :, j] + noise.flatten()
        return noise_S

    def get_Ci_f(self):
        assert self.o_fre < 9 * self.num_D, "o_fre should be less than 10 times num_D"

        Ci_f = np.zeros((self.num_D, self.n_fre))
        while True:
            num_fre = np.random.randint(0, 10, self.num_D)
            total_num_fre = np.sum(num_fre)
            if total_num_fre == self.o_fre:
                break

        chemical_shift = np.linspace(0, self.max_fre, self.n_fre)

        for i in range(self.num_D):
            if num_fre[i] == 0:  # Skip if no frequencies for this diffusion coefficient
                continue
            fre = np.random.rand(num_fre[i]) * self.max_fre
            lorz = self.sig_lorz ** 2 / ((chemical_shift[:, np.newaxis] - fre[np.newaxis, :]) ** 2 + self.sig_lorz ** 2)
            lorz_normalized = lorz / lorz.max(axis=0)
            Ci_f[i] = lorz_normalized.sum(axis=1)

        return Ci_f


def Gaussian_distribution(max_D, avg, num, sig):

    xgrid = np.linspace(0, max_D, num)
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((xgrid-avg),2))
    result = coef*(np.exp(mypow))

    if config.Type == 'T1T2':
        return result/np.max(result)
        
    return result/np.tile(np.max(result, axis=1).reshape(config.o_fre, 1), [1, config.label_size]) 

def gaussian_noise(S, dB):

    snr = np.exp(np.log(10) * float(dB) / 10)
    num_samples, n_fre, signal_dim = np.shape(S)
    noise_S = np.zeros([num_samples, n_fre, signal_dim])
    sigma = np.sqrt(1. / snr)
    weight_b = np.zeros([n_fre, signal_dim])
    weight_f = np.zeros([n_fre, signal_dim])

    snr_nmr = np.zeros([num_samples, n_fre])
    
    for i in trange(num_samples):
        noise = np.random.randn(n_fre, signal_dim)
        mult = sigma * np.linalg.norm(S[i, :, :], 2) / (np.linalg.norm(noise, 2))
        noise = noise * mult
        for j in np.arange(n_fre):
            # weight_b[j, :] = 1 / (np.arange(signal_dim) + 1)
            weight_b[j, :] = -np.linspace(0, 1, config.signal_dim) + 1
        for k in np.arange(signal_dim):
            weight_f[:, k] = S[i, :, 0] / np.max(S[i, :, 0], axis=0)

        if config.Type == 'T1T2':
            noise_S[i, :, :] = S[i, :, :] + noise
        else:
            noise_S[i, :, :] = S[i, :, :] + noise * weight_f
            for j in np.arange(n_fre):
                snr_nmr[i, j] = S[i,j,0]/mult/weight_f[j,0]

    return noise_S
def gen_signal_T1T2():

    num_samples = config.num_samples
    max_D = config.max_D
    label_size = config.label_size
    sig = config.sig
    signal_dim = config.signal_dim
    n_fre = 1
    min_sep = config.min_sep

    dB = config.dB

    b = (pow(2, np.arange(10)) * (config.max_b/(2**9)))

    S = np.zeros([num_samples, signal_dim, n_fre])
    label = np.zeros([num_samples, label_size, n_fre])

    for i in trange(num_samples):

        num_D = np.random.randint(config.num_D) + 1

        D = np.random.randint(0, num_D, [num_D, 1]).astype(float)

        if num_D == 1:
            D[D==0] = np.random.random(num_D) * max_D
            amp = 1
        else:
            while True:  # TODO
                D_value = np.random.random(num_D) * max_D
                D_value = D_value

                D_value = np.sort(D_value)
                D_value_t = np.roll(D_value, 1)

                if np.min(np.abs(D_value-D_value_t)) > min_sep:
                    break
            
            if n_fre == 1:
                for j in np.arange(num_D):
                    D[j] = D_value[j]
            else:
                for j in np.arange(num_D):
                    D[D==j] = D_value[j]

            # amp = (np.random.random() * 0.7) + 0.3
            # amp = np.array([[amp], [1-amp]])
            while True:
                amp = np.random.random([num_D, 1])
                if np.max(amp)/np.min(amp) < 3:
                    amp = amp/np.sum(amp)
                    break
        
        D = D + 2e-2
        signal = np.dot(np.exp(-b/D).T, amp)
        signal = signal/signal[0]

        S[i] = signal

        label[i] = np.sum(Gaussian_distribution(max_D, D, label_size, sig=sig), axis=0).reshape(config.label_size, 1)
        
    S = S.swapaxes(1, 2)
    label = label.swapaxes(1, 2)
    
    print('Add noise')
    noise_S = gaussian_noise(S, dB)

    return noise_S.astype('float32'), label.astype('float32')


if __name__ == '__main__':
    load_dataloader(config.batch_size)
