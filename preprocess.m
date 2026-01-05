close all;

data = load('data/P1S1.mat');
data = data.segdata;
%保留有效范围内的距离门
data = data(1:25, :);


L = size(data, 2);
fs = 20;
t = (0:length(data)-1)/fs;
channel = 1:1:25;

figure;
imagesc(t,channel,abs(data));
colorbar;
set(gca, 'YDir', 'normal');
xlabel('Time/s');ylabel('Channel');axis tight;
title("距离维FFT结果图")

target_bin = cfar(data);
select_data = data(target_bin,:);


phase_data = angle(select_data);
phase_data = unwrap(phase_data);

diff_phase = zeros(size(phase_data));
raw_diff = diff(phase_data);
diff_phase(2:end) = raw_diff;

figure;
plot(t, diff_phase);
xlabel('Time/s');ylabel('Phase');axis tight;
title("相位图")


data_fft = abs(fft(diff_phase)); 
data_fft(1:floor(0.1/(fs/L))) = 0;

f = (0:1:L/2-1)*fs/L;
figure;
plot(f(1:floor(2/(fs/L))),data_fft(1:floor(2/(fs/L))),'b-');xlabel('f/Hz');ylabel('Amplitude');
axis tight; title("信号的功率谱(0.1-2Hz)")


fs = 20;
nfft = 1024; % DFT 点数
window = hamming(145);
overlap = 144;



[S,F,T,P] = spectrogram(diff_phase(1:400), window, overlap, nfft, fs);
P(1:floor(0.1/(fs/nfft)),:) = 0;

figure;
imagesc(T,F(floor(0.1/(fs/nfft)):floor(2/(fs/nfft))),10*log10(P(floor(0.1/(fs/nfft)):floor(2/(fs/nfft)),:)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram');

function maxVarianceRow = cfar(data)
    rowVariances = var(data, [], 2);
    [~, maxVarianceRow] = max(rowVariances);
end