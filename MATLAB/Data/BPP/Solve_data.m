clc;clear;close all;

load BPP2.mat
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

DOSYData = DOSYData(:, 400:1000);
DOSYData = real(DOSYData);
DOSYData = DOSYData / max(DOSYData(:));

cat = 17;
DOSYData = DOSYData(1:cat,:);
b = b(1:cat);

[peaks, idx_peaks] = findpeaks(DOSYData(1, :), 'MinPeakHeight', 0.05, 'MinPeakProminence',0.001);
ppm = cs;
ppm = ppm(400:1000);
S = DOSYData(:, idx_peaks);
whole_spec = zeros([length(b), length(ppm)]);
whole_spec(:, idx_peaks) = S;
HNMR = DOSYData(1, idx_peaks)';
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

save('BPP2_net_input.mat', 'S', 'ppm', 'b', 'idx_peaks',"HNMR", '-mat');