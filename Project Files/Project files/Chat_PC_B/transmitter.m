% COMPLETE THE TRANSMITTER!

% pack = message to be transmitted (consists of 432 bits from the GUI, always!)
% fc = carrier frequency

%To test run: transmitter(randi(2,1,432)-1,1000)
function transmitter(pack,fc)

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
%preamble = [1 1 1 1 1 0 0 1 1 0 1 0 1];    %length 13 barker
%preamble = [1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 1 0 0 1 0];   %preamble to be used -  2x 11 BC
preamble = kron([1 1 0 1],[1 1 1 1 1 0 0 1 1 0 1 0 1]);
%preamble = [1 1 1 1 1 0 0 1 1 0 1 0 1];
%preamble = [preamble preamble];

Rb = 480;
Rs = Rb/bpsymb;
fsfd = fs/Rs;
Ts = 1/Rs;

%Root raised cosine pulse
[pulse, t] = rtrcpuls(alpha,Ts,fs,span);


pack = [preamble,pack'];            %prepend the preamble to the given pack of data
m = buffer(pack, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
m_idx = bi2de(m, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
x = const(m_idx);                   % Look up symbols using the indices
x_upsample = upsample(x,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
s = conv(pulse,x_upsample);         %Baseband signal to transmit
length(s)

%figure;
%plot(real(s))

%Modulated tx signal
%tx_signal = s.*exp(-1i*2*pi*fc*(0:length(s)-1)*Tsamp); % Carrier Modulation/Upconversion 
%tx_signal = real(tx_signal);                        % send real part, information is in amplitude and phase
%tx_signal = tx_signal/max(abs(tx_signal));          % Limit the max amplitude to 1 to prevent clipping of waveforms

%Modulate the baseband signal
tx_signal = s.*exp(-1i*2*pi*fc*(0:length(s)-1)*Tsamp); % Carrier Modulation/Upconversion 
tx_signal = real(tx_signal);                        % send real part, information is in amplitude and phase
tx_signal = tx_signal/max(abs(tx_signal));          % Limit the max amplitude to 1 to prevent clipping of waveforms

player = audioplayer(tx_signal, fs,16,4);       %create an audioplayer object to play the noise at a given sampling frequency
playblocking(player); % Play the noise 
