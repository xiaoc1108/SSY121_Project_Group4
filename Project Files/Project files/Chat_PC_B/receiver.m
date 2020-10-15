function [audio_recorder] = receiver(fc)
fs = 48000/2; %sampling frequency
audio_recorder = audiorecorder(fs,16,1,1);% create the recorder
audio_recorder.UserData.counter = 1; %initialize a counter in the structure UserData
audio_recorder.UserData.fc = fc;        %carrier freq
audio_recorder.UserData.fs = 48000/2;        %sample freq
audio_recorder.UserData.trigger = 0;
audio_recorder.UserData.maxCorr = 1;
audio_recorder.UserData.corrIdx = 1;
audio_recorder.UserData.Rb = 480;       %bitrate
audio_recorder.UserData.bpsymb = 2;     %bits/symb
audio_recorder.UserData.alpha = 0.4;    %roll off factor
audio_recorder.UserData.preambleBits = kron([1 1 0 1],[1 1 1 1 1 0 0 1 1 0 1 0 1]);    %kroniker product B4xB13 = length 52
audio_recorder.UserData.const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);            %qpsk constellation
audio_recorder.UserData.sPreamble = [];
audio_recorder.UserData.corrAng = [];



%attach callback function
time_value = 0.2; % how often the function should be called in seconds
set(audio_recorder,'TimerPeriod',time_value,'TimerFcn',@audioTimerFcn); % attach a function that should be called every second, the function that is called is specified below.

%ADD USER DATA FOR CALLBACK FUNCTION (DO NOT CHANGE THE NAMES OF THESE VARIABLES!)
audio_recorder.UserData.receive_complete = 0; % this is a flag that the while loop in the GUI will check
audio_recorder.UserData.pack  = []; %allocate for data package
audio_recorder.UserData.pwr_spect = []; %allocate for PSD
audio_recorder.UserData.const = []; %allocate for constellation
audio_recorder.UserData.eyed  = []; %allocate for eye diagram


record(audio_recorder); %start recording
end


% CALLBACK FUNCTION
% This function will be called every [time_value] seconds, where time_value
% is specified above. Note that, as stated in the project MEMO, all the
% fields: pwr_spect, eyed, const and pack need to be assigned if you want
% to get outputs in the GUI.

function audioTimerFcn(recObj, event, handles)

if recObj.UserData.trigger == 0
    rx = getaudiodata(recObj);  %recorded data
    rx = rx/max(abs(rx));       %normalize

    fs = recObj.UserData.fs;
    fc = recObj.UserData.fc;
    Rb = recObj.UserData.Rb;           %bit rate
    Rs = Rb/recObj.UserData.bpsymb;    %Symbol rate
    fsfd = fs/Rs;                      %samples per symbol
    Ts = 1/Rs;                         %symbol time
    Tsamp = 1/fs;
    alpha = recObj.UserData.alpha;
    span = 6;

    %Make RRC pulse
    [pulse, t] = rtrcpuls(alpha,Ts,fs,span);

    %Down modulate rx signal
    rxBaseband = rx'.*exp(-1i*2*pi*fc*(0:length(rx')-1)*Tsamp);
    
    %Match filter
    mfOutput = matchFilter(pulse, rxBaseband);

    %correlate received signal and preamble
    sPreamble = makePreamble(pulse,fsfd,2);
    [pwrRx, f] = pwelch(mfOutput,1024,768,1024, fs);
    [pwrPre, f] = pwelch(sPreamble,1024,768,1024, fs);
    corr = conv((mfOutput), fliplr(sPreamble));
    corr = corr/(sum(pwrRx/max(pwrRx))*sum(pwrPre/max(pwrPre))); %normalize by preamble and signal pwr
    
    %Find point of max correlation   
    [maxCorr,maxIdx] = max(abs(corr));
    
    recObj.UserData.corrIdx = maxIdx;
    recObj.UserData.maxCorr = maxCorr;
    recObj.UserData.sPreamble = sPreamble;
    recObj.UserData.corrAng = angle(corr(maxIdx));

    
    if recObj.UserData.maxCorr > 15     %If the max peak excedes the threshold of 15
        fprintf('Triggered ---->')
        recObj.UserData.trigger = 1;    %Set the trigger flag when threshold is hit
    end
elseif recObj.UserData.counter < 6      %Wait for whole signal to be recorded
            
    recObj.UserData.counter = recObj.UserData.counter + 1;
            
elseif recObj.UserData.counter == 6
    fprintf('Stopped ----->')
    stop(recObj)
    
    rx = getaudiodata(recObj);  %recorded data
    rx = rx/max(abs(rx));

    fs = recObj.UserData.fs;
    fc = recObj.UserData.fc;
    Rb = recObj.UserData.Rb;
    Rs = Rb/recObj.UserData.bpsymb;
    fsfd = fs/Rs;
    Ts = 1/Rs;
    Tsamp = 1/fs;
    alpha = recObj.UserData.alpha;
    span = 6;

    %Make RRC pulse
    [pulse, t] = rtrcpuls(alpha,Ts,fs,span);

    %Down modulate rx signal
    rxBaseband = rx'.*exp(-1i*2*pi*fc*(0:length(rx')-1)*Tsamp);
    
    %Match filter
    mfOutput = matchFilter(pulse, rxBaseband);

    %correlate received signal and preamble
    sPreamble = makePreamble(pulse,fsfd,2);

    %Find delay value for start of preamble
    delay = recObj.UserData.corrIdx - length(sPreamble) + span*fsfd;

    %Clip the received signal
    val = 25398;                %length of tx message
    rxClipped = mfOutput(delay:val+delay-2*(span*fsfd));

    %Multiply the rx signal by phase of correlation to correct the signal
    %when there is a large imaginary spike in the correlation.
    rxClipped = rxClipped*exp(-1i*recObj.UserData.corrAng);

    %Downsample the rx signal to the actual received symbols
    rxDown = conj(downsample(rxClipped,fsfd));

    %Symbol phase correction
    [sPreamble,xPreamble] = makePreamble(pulse,fsfd,2);
    %Find all rxVec points in 1+1i quadrant (upper right)
    I1Preamb = find(real(xPreamble) > 0 & imag(xPreamble) > 0);
    I2Preamb = find(real(xPreamble) < 0 & imag(xPreamble) > 0);
    I3Preamb = find(real(xPreamble) < 0 & imag(xPreamble) < 0);
    I4Preamb = find(real(xPreamble) > 0 & imag(xPreamble) < 0);

    phase1 = mean(angle(rxDown(I1Preamb)))*180/pi;
    phase2 = mean(angle(rxDown(I2Preamb)))*180/pi;
    phase3 = mean(angle(rxDown(I3Preamb)))*180/pi;
    phase4 = mean(angle(rxDown(I4Preamb)))*180/pi;
    deltaPhase1 = phase1-45;
    deltaPhase2 = phase2-135;
    deltaPhase3 = phase3+135;
    deltaPhase4 = phase4+45;
    deltaPhase = (deltaPhase1+deltaPhase2+deltaPhase3+deltaPhase4)/4;
    rxDown = rxDown*exp(-1i*deltaPhase*pi/180);


    const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);
    eucDist = abs(repmat(rxDown.',1,4) - repmat(const, length(rxDown), 1)).^2;
    [~,mHat] = min(eucDist, [], 2);
    rxBitsBuffer = de2bi(mHat'-1, 2, 'left-msb')'; %make symbols into bits
    rxBits = rxBitsBuffer(:)'; %write as a vector

    % Step 1: save the estimated bits
    recObj.UserData.pack = rxBits(53:end);

    % Step 2: save the sampled symbols
    recObj.UserData.const = rxDown(27:end)/max(abs(rxDown(27:end)));

    % Step 3: provide the matched filter output for the eye diagram
    recObj.UserData.eyed.r = rxClipped;
    recObj.UserData.eyed.fsfd = fsfd;

    % Step 4: Compute the PSD and save it. 
    % !!!! NOTE !!!! the PSD should be computed on the BASE BAND signal BEFORE matched filtering
    [pxx, f] = pwelch(rxBaseband,1024,768,1024, fs); % note that pwr_spect.f will be normalized frequencies
    f = fftshift(f); %shift to be centered around fs
    f(1:length(f)/2) = f(1:length(f)/2) - fs; % center to be around zero
    p = fftshift(10*log10(pxx/max(pxx))); % shift, normalize and convert PSD to dB
    recObj.UserData.pwr_spect.f = f;
    recObj.UserData.pwr_spect.p = p;

    % In order to make the GUI look at the data, we need to set the
    % receive_complete flag equal to 1:
    disp('Done')
    recObj.UserData.trigger = 0;
    recObj.UserData.receive_complete = 1;
end
end

function MF_output_conv = matchFilter(pulse, signal)
    MF = fliplr(conj(pulse));        %create matched filter impulse response
    MF_output_conv = conv(pulse, signal);
    MF_output_conv = MF_output_conv(length(MF):end-length(MF)+1);   %remove the transients from the MF
end

function [preambleTrain,xPreamble] = makePreamble(pulse,samplesSymb,bitsSymb)
    preamble = kron([1 1 0 1],[1 1 1 1 1 0 0 1 1 0 1 0 1]);
    const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);     %qpsk constelation
    mPreamble = buffer(preamble, bitsSymb)';            %Group 2 bits per symbol (each row will be a symbol)
    mPreIdx = bi2de(mPreamble, 'left-msb')'+1;              % Bits to symbol index, msb: the Most Significant Bit
    xPreamble = const(mPreIdx);                             % Look up symbols using the indices
    xPreUpsample = upsample(xPreamble,samplesSymb);         % Space the symbols fsfd apart, to enable pulse shaping using conv.
    preambleTrain = conv(pulse,xPreUpsample);               %The baseband pulse shaped preamble sequency (ask about this in the Q&A)
end
