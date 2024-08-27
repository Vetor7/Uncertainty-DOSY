clc;clear;close all;

load JNN.mat

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
DOSYData = DOSYData(1:11,:);
b = b(1:11);
[peaks, idx_peaks] = findpeaks(DOSYData(1, :), 'MinPeakHeight', 0.03, 'MinPeakProminence',0.01);
% idx_peaks = find(DOSYData(1, :) >= 0.0677);
ppm = cs;

S = (DOSYData(:, idx_peaks) + DOSYData(:, idx_peaks-1) + DOSYData(:, idx_peaks+1))/3;
HNMR = DOSYData(1, idx_peaks)';
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

save('JNN_net_input.mat', 'S', 'ppm', 'b', 'idx_peaks', 'HNMR', '-mat');