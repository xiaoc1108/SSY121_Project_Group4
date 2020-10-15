function transmitter(pack,fc)

fs = 48000;                                     %sampling frequency
B = 200;                                        %1 sided bandwidth [hz]
Tsamp = 1/fs;                                   %sample time
alpha = 0.4;                                    %rolloff factor for rrc pulse
const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);     %qpsk - 2 bits per symbol
M = length(const);                              %number of symbols (2^2)
bpsymb = log2(M);                               %bits per symbol
span = 6;

preamble = kron([1 1 0 1],[1 1 1 1 1 0 0 1 1 0 1 0 1]);

Rb = 480;                                       %Bit rate
Rs = Rb/bpsymb;                                 %Symbol rate
fsfd = fs/Rs;                                   %samples per symbol
Ts = 1/Rs;                                      %symbol time

%Root raised cosine pulse
[pulse, t] = rtrcpuls(alpha,Ts,fs,span);


pack = [preamble,pack'];            %prepend the preamble to the given pack of data
m = buffer(pack, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
m_idx = bi2de(m, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
x = const(m_idx);                   % Look up symbols using the indices
xUpsample = upsample(x,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
s = conv(pulse,xUpsample);         %Baseband signal to transmit

%Modulate the baseband signal
txSignal = s.*exp(-1i*2*pi*fc*(0:length(s)-1)*Tsamp); % Carrier Modulation/Upconversion 
txSignal = real(txSignal);                        % send real part, information is in amplitude and phase
txSignal = txSignal/max(abs(txSignal));          % Limit the max amplitude to 1 to prevent clipping of waveforms

player = audioplayer(txSignal, fs,16,3);       %create an audioplayer object to play the noise at a given sampling frequency
playblocking(player); % Play the noise 
