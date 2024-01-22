clc;clear;close all;

load QGC.mat

DOSYData = NmrData.SPECTRA;% DOSYData    
 if size(DOSYData,1)>size(DOSYData,2)
     DOSYData = DOSYData.';  
 end

g=100*NmrData.Gzlvl; % gradient values
BD=NmrData.DELTAOriginal; % diffusion time
LD=NmrData.deltaOriginal; % diffusion encoding time
cs=NmrData.Specscale;     % chemical shift
gamma = 4257.7;
g2 = (2*pi*gamma*g*LD).^2*(BD-LD/3)*1e4;
b = g2*1e-10;

DOSYData = real(DOSYData);
DOSYData = DOSYData / max(DOSYData(:));

idx_peaks = find(DOSYData(1, :) >= 0.01377);
ppm = cs;

S = DOSYData(:, idx_peaks);
% 设置所需的信噪比（单位：分质数 dB）
desired_SNR_dB = 30; % 例如 20 dB

% 计算原始信号 S 的功率
signal_power = var(S(:));

% 将所需的 SNR 从 dB 转换为线性比率
desired_SNR_linear = 10^(desired_SNR_dB / 10);

% 计算噪声的功率
noise_power = signal_power / desired_SNR_linear;

% 生成高斯噪声
noise = sqrt(noise_power) * randn(size(S));

% 只在选定的区域添加噪声
S_noisy = S;
S(:, 2000:2277) = S(:, 2000:2277) + noise(:, 2000:2277);
% S(:, :) = S(:, :) + noise(:,:);

whole_spec = zeros([length(b), length(ppm)]);
whole_spec(:, idx_peaks) = S;

figure(1)
subplot(2, 1, 1)
plot(ppm, DOSYData(1, :))
hold on
subplot(2, 1, 2)
plot(ppm, whole_spec(1, :))
S_new = S ./ S(1, :);

figure(2)
plot(b, S_new(:, :))
S = S';

save('QGC_net_input.mat', 'S', 'ppm', 'b', 'idx_peaks', '-mat');