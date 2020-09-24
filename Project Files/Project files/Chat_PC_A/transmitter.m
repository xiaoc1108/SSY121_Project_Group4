% COMPLETE THE TRANSMITTER!

% pack = message to be transmitted (consists of 432 bits from the GUI, always!)
% fc = carrier frequency

%To test run: transmitter(randi(2,1,432)-1,1000)
function transmitter(pack,fc)

fs = 44100;                                     %sampling frequency
B = 200;                                        %1 sided bandwidth [hz]
Tsamp = 1/fs;                                   %sample time
alpha = 0.4;                                    %rolloff factor for rrc pulse
G = (1+alpha)/(2*B);                            %Arbitrary paramater
k = 2;                                          %integer multiple
Ts = k*G;                                       %symbol time (for a root raised cosine)
fsymb = 1/Ts;                                   %symbol rate [symb/s]
const = [(1+1i), (1-1i), (-1-1i), (-1+1i)];     %qpsk - 2 bits per symbol
M = length(const);                              %number of symbols (2^2)
bpsymb = log2(M);                               %bits per symbol
Rb = round(fsymb*bpsymb);                       %bit rate [bit/s]
fsfd = round(fs/fsymb)+1;                           %samples per symbol
span = 6;

%Implement root raised cosine pulse
t_positive = eps:(1/fs):span*Ts;  % Replace 0 with eps (smallest +ve number MATLAB can produce) to prevent NANs
t = [-fliplr(t_positive(2:end)) t_positive];
tpi = pi/Ts; amtpi = tpi*(1-alpha); aptpi = tpi*(1 + alpha);
ac = 4*alpha/Ts; at = 16*alpha^2/Ts^2;
pulse = (sin(amtpi*t) + (ac*t).*cos(aptpi*t))./(tpi*t.*(1-at*t.^2));
pulse = pulse/norm(pulse);

%n = length(pulse);
%Y = fftshift(fft(pulse));
%fshift = (-n/2:n/2-1)*(fs/n);
%powershift = abs(Y).^2/n;     % zero-centered power
%plot(fshift,powershift)
%figure; stem(t,pulse);

%pack(401:end)
m = buffer(pack, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
m_idx = bi2de(m, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
x = const(m_idx);                   % Look up symbols using the indices
x_upsample = upsample(x,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
s = conv(pulse,x_upsample);         %Baseband signal to transmit

%
%Plots to loot at baseband transmitted signal
figure; 
subplot(2,1,1); 
plot(Tsamp*(0:(length(s)-1)), real(s), 'b');
samples = s(span*fsfd:fsfd:end-span*fsfd);
t = span*fsfd:fsfd:(span*fsfd + fsfd*(length(samples)-1));
hold on;
stem(t*Tsamp, real(samples),'r');
legend('s', 'sampled s');
title('real s')
xlabel('seconds')

%subplot(2,1,2); 
%plot(Tsamp*(0:(length(s)-1)), imag(s), 'b');
%hold on;
%stem(t*Tsamp, imag(samples),'r');
%legend('s', 'sampled s');
%title('imag')
%xlabel('seconds')

subplot(2,1,2)
stem(real(x))
title('Real X')
%}

tx_signal = s.*exp(-1i*2*pi*fc*(0:length(s)-1)*Tsamp); % Carrier Modulation/Upconversion 
tx_signal = real(tx_signal);                        % send real part, information is in amplitude and phase
SNR = 20;   %noise in db
tx_signal = tx_signal/max(abs(tx_signal));
noisy_tx_signal = awgn(tx_signal,SNR,'measured');          % Limit the max amplitude to 1 to prevent clipping of waveforms

%
baseband_signal = noisy_tx_signal.*exp(-1i*2*pi*fc*(0:length(tx_signal)-1)*Tsamp);
%baseband_signal = real(baseband_signal);
MF = fliplr(conj(pulse));
MF_output_conv = conv(pulse, baseband_signal);  % Another approach to MF using conv, what's the difference?
MF_output_conv = MF_output_conv(floor(length(MF)):end-floor(length(MF)+1));
MF_output = filter(MF,1,baseband_signal);
MF_output = MF_output(length(MF):end);
rx_vec = MF_output_conv(1:fsfd:end);  %get sample points
scatterplot(rx_vec); %scatterplot of received symbols
%size(MF)
%size(MF_output_conv)
%size(baseband_signal)
%size(rx_vec)
%rx_vec
%scatterplot(x);
%
figure
subplot(2,1,1)
plot(real(s))
subplot(2,1,2)
plot(real(MF_output_conv));
%}
%{
figure
plot(tx_signal)

N = 256;
P_bb = fftshift(fft(s, N)); % baseband signal
P_bp = fftshift(fft(tx_signal, N)); % passband signal
df = fs/N;                  % sampling frequency is split into N bins
fvec = df*(-floor(N/2):1:ceil(N/2)-1); % Truncated, has wide bandwidth
figure; plot(fvec, 20*log10(abs(P_bb/N))); hold on
plot(fvec, 20*log10(abs(P_bp/N))); legend('Baseband signal', 'Passband signal')
%}
%tx_signal = randn(1,1e4); %vector of noise
player = audioplayer(tx_signal, fs);       %create an audioplayer object to play the noise at a given sampling frequency
playblocking(player); % Play the noise 
