%comment
clear;
close all;
N = 432;
%rng(0,'twister');
%preamble = [1 1 1 1 1 0 0 1 1 0 1 0 1];
%preamble = [1 1 1 0 0 0 1 0 0 1 0];
%preamble = [1 1 0 1 1 0 1 1 0 1 1 0 1 1 1 0 0 1]; %Some random preamble I was trying out
preamble = [1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 1 0 0 1 0];
%preamble = [1 1 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1]; %2 13 barkers
%preamble = [1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1];
data = randsrc(1,N,[0 1]);
pack = [preamble,data];
%crap = awgn(randsrc(1,1234,[-1 1]),-5,'measured'); %this is the useless signal we can put before the pack at the receiver end
fc = 2500;

fs = 48000;                                     %sampling frequency
B = 200;                                        %1 sided bandwidth [hz]
Tsamp = 1/fs;                                   %sample time
alpha = 0.4;                                    %rolloff factor for rrc pulse
G = (1+alpha)/(2*B);                            %Arbitrary paramater
k = 1;                                          %integer multiple
Ts = k*G;                                       %symbol time (for a root raised cosine)
fsymb = 1/Ts;                                   %symbol rate [symb/s]
const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);     %qpsk - 2 bits per symbol
M = length(const);                              %number of symbols (2^2)
bpsymb = log2(M);                               %bits per symbol
Rb = round(fsymb*bpsymb);                       %bit rate [bit/s]
fsfd = round(fs/fsymb)+1;                           %samples per symbol
span = 6;

Rb = 480;
Rs = Rb/bpsymb;
fsfd = fs/Rs;
Ts = 1/Rs;

%This will be the noisy stuff before the preamble so we can see if our
%system detects the preamble
crapIdx = randi(4,1050,1)';
crap = awgn(const(crapIdx),-5,'measured');

%Implement root raised cosine pulse
t_positive = eps:(1/fs):span*Ts;  % Replace 0 with eps (smallest +ve number MATLAB can produce) to prevent NANs
t = [-fliplr(t_positive(2:end)) t_positive];
tpi = pi/Ts; amtpi = tpi*(1-alpha); aptpi = tpi*(1 + alpha);
ac = 4*alpha/Ts; at = 16*alpha^2/Ts^2;
pulse = (sin(amtpi*t) + (ac*t).*cos(aptpi*t))./(tpi*t.*(1-at*t.^2));
pulse = pulse/norm(pulse);

%[pulse, t] = rtrcpuls(alpha,Ts,fs,span);

%pack(401:end)
m = buffer(pack, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
m_idx = bi2de(m, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
x = const(m_idx);                   % Look up symbols using the indices
x_upsample = upsample(x,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
s = conv(pulse,x_upsample);         %Baseband signal to transmit

%{
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

subplot(2,1,2)
stem(real(x))
title('Real X')
%}

%Modulated tx signal
tx_signal = s.*exp(-1i*2*pi*fc*(0:length(s)-1)*Tsamp); % Carrier Modulation/Upconversion 
tx_signal = real(tx_signal);                        % send real part, information is in amplitude and phase
tx_signal = tx_signal/max(abs(tx_signal));          % Limit the max amplitude to 1 to prevent clipping of waveforms


%From here on it is like receiver stuff
SNR =0;   %signal to noise ratio
rx_signal = awgn(tx_signal,SNR,'measured');          
%rx_signal = tx_signal;
rxBaseband = rx_signal.*exp(-1i*2*pi*fc*(0:length(s)-1)*Tsamp); %down modulate
rxBaseband = [crap,rxBaseband]; %add the nonsense signal before the rx_baseband


%Add phase error to the received signal
phaseError = 33; %in degrees
theta = phaseError*(pi/180);
rxBaseband = rxBaseband*exp(-1i*theta);


%Make baseband preamble sequence
mPreamble = buffer(preamble, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
mPreIdx = bi2de(mPreamble, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
xPreamble = const(mPreIdx);                   % Look up symbols using the indices
xPreUpsample = upsample(xPreamble,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
sPreamble = conv(pulse,xPreUpsample);         %The baseband pulse shaped preamble sequency (ask about this in the Q&A)

corr = conv((rxBaseband), fliplr(sPreamble));   % correlate the sequence and rx_baseband
corr = normalize(corr);
[tmp, Tmax] = max(abs(corr));         %find location of max correlation (this should be where the preamble starts)
%fprintf('delay = %d \n',Tmax-length(sPreamble));
delay = Tmax - length(sPreamble) %this point will be where the preamble starts, so for rx_baseband we will take from this point onwards?
if delay < 0
    delay = 0;
end

rxBaseband = rxBaseband(delay+3:end); %cut out the useless signal from rx_baseband (also ask if this is correct way)

%phasePreamble = angle(sPreamble);
%phaseRxPreamble = angle(rxBaseband(1:length(sPreamble)));



MF = fliplr(conj(pulse));        %create matched filter impulse response
MF_output = filter(MF,1,rxBaseband);      % run received signal through matched filter
%figure; plot(real(MF_output))
MF_output = MF_output(length(MF):end); %remove transient
MF_output_conv = conv(pulse, rxBaseband);  % Another approach to MF using conv, what's the difference?
%figure; plot(real(MF_output))
MF_output_conv = conj(MF_output_conv(length(MF):end-length(MF)+1));
rxVec = MF_output_conv(1:fsfd:end);  %get sample points


%Symbol phase correction

%Find all rxVec points in 1+1i quadrant (upper right)
I1Preamb = find(real(xPreamble) > 0 & imag(xPreamble) > 0);
I2Preamb = find(real(xPreamble) < 0 & imag(xPreamble) > 0);
I3Preamb = find(real(xPreamble) < 0 & imag(xPreamble) < 0);
I4Preamb = find(real(xPreamble) > 0 & imag(xPreamble) < 0);

phase1 = mean(angle(rxVec(I1Preamb)))*180/pi;
phase2 = mean(angle(rxVec(I2Preamb)))*180/pi;
phase3 = mean(angle(rxVec(I3Preamb)))*180/pi;
phase4 = mean(angle(rxVec(I4Preamb)))*180/pi;
deltaPhase1 = phase1-45;
deltaPhase2 = phase2-135;
deltaPhase3 = phase3+135;
deltaPhase4 = phase4+45;
deltaPhase = (deltaPhase1+deltaPhase2+deltaPhase3+deltaPhase4)/4;
rxVec = rxVec*exp(-1i*deltaPhase*pi/180);

%{
figure
sgtitle('Real')
subplot(3,1,1)
plot(real(s))
title('Baseband transmitted signal')
subplot(3,1,2)
plot(real(awgn(s,SNR,'measured')))
title('Noisy baseband transmitted signal')
subplot(3,1,3)
plot(real(MF_output_conv));
title('Received output after matched filtering')
figure
sgtitle('Imaginary')
subplot(3,1,1)
plot(imag(s))
title('Baseband transmitted signal')
subplot(3,1,2)
plot(imag(awgn(s,SNR,'measured')))
title('Noisy baseband transmitted signal')
subplot(3,1,3)
plot(imag(MF_output_conv));
title('Received output after matched filtering')
%}

%Symbols to bits
eucDist = abs(repmat(rxVec.',1,4) - repmat(const, length(rxVec), 1)).^2;
[tmp mHat] = min(eucDist, [], 2);
rxSymbols = const(mHat);

SER = symerr(m_idx, mHat') %count symbol errors (sometimes errors when the fram synchronization is 1 or 2 positions off, how can we never have errors?)
%SER = symerr(mDataIdx, mHat') %count symbol errors
%mDataIdx
rxBitsBuffer = de2bi(mHat'-1, 2, 'left-msb')'; %make symbols into bits
rxBits = rxBitsBuffer(:)'; %write as a vector
BER = biterr(pack, rxBits) %count of bit errors

%player = audioplayer(tx_signal, fs);       %create an audioplayer object to play the noise at a given sampling frequency
%playblocking(player); % Play the noise 


%Plots

figure; plot(abs(corr), '.-r'); title('Correlation between rx_{baseband} and preamble pulse sequence')       % plot correlation

scatterplot(rxVec/max(abs(rxVec))); %scatterplot of received symbols
%{
figure
subplot(3,1,1)
plot(Tsamp*(0:(length(sPreamble)-1)), real(sPreamble), 'b');
samples = sPreamble(span*fsfd:fsfd:end-span*fsfd);
t = span*fsfd:fsfd:(span*fsfd + fsfd*(length(samples)-1));
hold on;
stem(t*Tsamp, real(samples),'r');
subplot(3,1,2)
plot(real(sPreamble(span*fsfd:end-span*fsfd)));
title('Using span*fsfd')
subplot(3,1,3)
plot(real(sPreamble(round(length(pulse)/2):end-round(length(pulse)/2))));
title('Using pulse/2')

figure
plot(real(rxVec))
%}

%%

fs = 44100 ; 
nBits = 24 ;  
ID = -1; % default audio input device 
%recObj = audiorecorder(Fs,nBits,1);
%disp('Start speaking.')
%recordblocking(recObj,5);
%disp('End of Recording.');
%play(recObj);

 %covert to floating-point vector and play back
yi = getaudiodata(recObj,'int16');
plot(yi)
outData = bitshift(yi, -1);
%sound(y,fs);if true
%end
