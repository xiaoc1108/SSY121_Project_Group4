% RECEIVER 
%
% This is the receiver structure that you will have to complete.
% The function: receiver(fc) is a setup function for the receiver. Here,
% the audiorecorder object is initialized (see help audiorecorder or
% MATLAB's homepage for more information about the object).
% 
% The callback function audioTimerFcn() is a callback function that is
% triggered on a specified time interval (here it is determined in the
% setup function, by the variable time_value)
% 
% Your task is to extend this code to work in the project!
%%

function [audio_recorder] = receiver(fc)
%fc = 3000;
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
audio_recorder.UserData.preambleBits = kron([1 1 0 1],[1 1 1 1 1 0 0 1 1 0 1 0 1]);    %kroniker product B4xB13 = length 52
audio_recorder.UserData.const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);
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

% So, let's see an example of where we have a pulse train as in Computer
% exercise 2 and let the GUI plot it. Note that we will just create 432
% random bits and hence, the GUI will not be able to decode the message but
% only to display the figures.
% Parameters in the example:
% f_s = 22050 [samples / second]
% R_s = 350 [symbols / second]
% fsfd = f_s/R_s [samples / symbol]
% a = 432 bits
% M = 4 (using QPSK as in computer exercise)

function audioTimerFcn(recObj, event, handles)

%if recObj.UserData.counter < 14
%   recObj.UserData.counter = recObj.UserData.counter + 1;
%else
    %stop(recObj)
    %disp('Callback')
    %50798 - b4xb13
if recObj.UserData.trigger == 0
    rx = getaudiodata(recObj);  %recorded data
    rx = rx/max(abs(rx));

    fs = recObj.UserData.fs;
    fc = recObj.UserData.fc;
    Rb = recObj.UserData.Rb;
    Rs = Rb/recObj.UserData.bpsymb;
    fsfd = fs/Rs;
    Ts = 1/Rs;
    Tsamp = 1/fs;
    alpha = 0.4;
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
    %figure; subplot(2,1,1); plot(abs(corr)); subplot(2,1,2); plot(angle(corr))
   
%     figure;
%     subplot(2,1,1)
%     plot(real(corr))
%     subplot(2,1,2)
%     plot(imag(corr))
    
    %Find point of max correlation   
    [maxCorr,maxIdx] = max(abs(corr));
    
    recObj.UserData.corrIdx = maxIdx;
    recObj.UserData.maxCorr = maxCorr;
    recObj.UserData.sPreamble = sPreamble;
    recObj.UserData.corrAng = angle(corr(maxIdx));

    
    if recObj.UserData.maxCorr > 15     %maybe this can be increased
        fprintf('Triggered ---->')
        %recObj.UserData.corrIdx = maxIdx;
        %recObj.UserData.maxCorr = maxCorr;
        %recObj.UserData.sPreamble = sPreamble;
        %recObj.UserData.corrAng = angle(corr(maxIdx));
        recObj.UserData.trigger = 1;    %Set the trigger flag when threshold is hit
    end
elseif recObj.UserData.counter < 8
            
    recObj.UserData.counter = recObj.UserData.counter + 1;
            
elseif recObj.UserData.counter == 8
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
    alpha = 0.4;
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
    %figure; subplot(2,1,1); plot(abs(corr)); subplot(2,1,2); plot(angle(corr))
   
%     figure;
%     subplot(2,1,1)
%     plot(real(corr))
%     subplot(2,1,2)
%     plot(imag(corr))
    
    %Find point of max correlation   
    [maxCorr,maxIdx] = max(abs(corr)); 

    %Find delay value for start of preamble
    delay = maxIdx - length(sPreamble) + span*fsfd;

    %Clip the received signal
    val = 25398;
    %val = 50798;
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

    %scatterplot(rxDown)

    const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);
    eucDist = abs(repmat(rxDown.',1,4) - repmat(const, length(rxDown), 1)).^2;
    [maxCorr,mHat] = min(eucDist, [], 2);
    %rxSymbols = const(mHat);
    rxBitsBuffer = de2bi(mHat'-1, 2, 'left-msb')'; %make symbols into bits
    rxBits = rxBitsBuffer(:)'; %write as a vector

    %sum(rxBits(1:26) == [1 1 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1])

%     figure
%     subplot(2,1,1)
%     plot(real(mfOutput))
%     subplot(2,1,2)
%     plot(real(rxClipped))

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
    recObj.UserData.receive_complete = 1;
end
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kind of works
% if recObj.UserData.trigger == 0
%     %disp('Callback triggered')
%     rec_data = 0;
%     rxBaseband = 0;
%     rec_data = getaudiodata(recObj);
% 
%     %Setup info
%     %fs = 44100; %sampling frequency
%     %fc = 1000;
%     fc = recObj.UserData.fc;
%     fs = recObj.UserData.fs;
%     B = 200;                                        %1 sided bandwidth [hz]
%     Tsamp = 1/fs;                                   %sample time
%     alpha = 0.4;                                    %rolloff factor for rrc pulse
%     G = (1+alpha)/(2*B);                            %Arbitrary paramater
%     k = 1;                                          %integer multiple
%     Ts = k*G;                                       %symbol time (for a root raised cosine)
%     fsymb = 1/Ts;                                   %symbol rate [symb/s]
%     const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);     %qpsk - 2 bits per symbol
%     M = length(const);                              %number of symbols (2^2)
%     bpsymb = log2(M);                               %bits per symbol
%     Rb = round(fsymb*bpsymb);                       %bit rate [bit/s]
%     fsfd = round(fs/fsymb)+1;                           %samples per symbol
%     span = 6;
% 
%     Rb = 480;               %bit rate
%     Rs = Rb/bpsymb;         %ssymbol rate
%     fsfd = fs/Rs;           %samples/symbol
%     Ts = 1/Rs;              %Symbol time
% 
%     %Root raised cosine pulse
%     [pulse, t] = rtrcpuls(alpha,Ts,fs,span);
% 
%     %Down modulate the received signal
%     rxBaseband = rec_data'.*exp(-1i*2*pi*fc*(0:length(rec_data')-1)*Tsamp); %down modulate
% 
%     %Make baseband preamble sequence
%     preamble = [1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 1 0 0 1 0];   %preamble to be used -  2x 11 BC
%     mPreamble = buffer(preamble, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
%     mPreIdx = bi2de(mPreamble, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
%     xPreamble = const(mPreIdx);                   % Look up symbols using the indices
%     xPreUpsample = upsample(xPreamble,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
%     sPreamble = conv(pulse,xPreUpsample);         %The baseband pulse shaped preamble sequency (ask about this in the Q&A)
% 
%     %Correlate the signal and preamble
%     corr = conv((rxBaseband), fliplr(sPreamble));
%     %figure(1); clf; plot(real(corr))
% 
%     [tmp, Tmax] = max(abs(real(corr)));
% 
%     if tmp > 1.5
%         recObj.Userdata.trigger = 1;
%         recObj.UserData.maxCorr = tmp;
%         recObj.UserData.corrIdx = Tmax;
%         disp('Triggered')
%     end
% elseif recObj.userData.trigger == 1 && recObj.UserData.counter < 5
%     recObj.UserData.counter = recObj.UserData.counter + 1;
%     
% elseif recObj.UserData.trigger == 1 && recObj.UserData.counter == 5
%     stop(recObj);
%     disp('Stopped');
%     rec_data = getaudiodata(recObj);
%     
%     %Setup info
%     %fs = 44100; %sampling frequency
%     %fc = 1000;
%     fc = recObj.UserData.fc;
%     fs = recObj.UserData.fs;
%     Tsamp = 1/fs;                                   %sample time
%     alpha = 0.4;                                    %rolloff factor for rrc pulse
%     const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);     %qpsk - 2 bits per symbol
%     M = length(const);                              %number of symbols (2^2)
%     bpsymb = log2(M);                               %bits per symbol
%     span = 6;
% 
%     Rb = 480;               %bit rate
%     Rs = Rb/bpsymb;         %ssymbol rate
%     fsfd = fs/Rs;           %samples/symbol
%     Ts = 1/Rs;              %Symbol time
%     
%     %Root raised cosine pulse
%     [pulse, t] = rtrcpuls(alpha,Ts,fs,span);
%     
%     %Down modulate
%     rxBaseband = rec_data'.*exp(-1i*2*pi*fc*(0:length(rec_data')-1)*Tsamp); %down modulate
%     rx = rxBaseband;
%     
%     %Make baseband preamble sequence
%     preamble = [1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 1 0 0 1 0];   %preamble to be used -  2x 11 BC
%     mPreamble = buffer(preamble, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
%     mPreIdx = bi2de(mPreamble, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
%     xPreamble = const(mPreIdx);                   % Look up symbols using the indices
%     xPreUpsample = upsample(xPreamble,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
%     sPreamble = conv(pulse,xPreUpsample);         %The baseband pulse shaped preamble sequency (ask about this in the Q&A)
% 
%     %Correlate the signal and preamble
%     corr = conv((rxBaseband), fliplr(sPreamble));
%     
%     %trying to invert the signal
%     if abs(min(real(corr))) > abs(max(real(corr)))
%         rxBaseband = rxBaseband*exp(1i*pi); %rotate by 180 degrees;
%         disp('Yes')
%     end
%     %Clip rxBaseband to the correct length
%     %[tmp, Tmax] = max(abs(real(corr)));         %find location of max correlation (this should be where the preamble starts)
%     delay = recObj.UserData.corrIdx - length(sPreamble);
%     %if delay < 0
%     %    delay = 0;
%     %end
%     rxBaseband = rxBaseband(delay+2:(47798)+delay+2);
%     
%     MF_output_conv = matchFilter(pulse,rxBaseband);
%     rxVec = (MF_output_conv(1:fsfd:end)); 
%     rxVec = (rxVec(1:227));
%     
%      %Symbol phase correction
%     %Find all rxVec points in 1+1i quadrant (upper right)
%     I1Preamb = find(real(xPreamble) > 0 & imag(xPreamble) > 0);
%     I2Preamb = find(real(xPreamble) < 0 & imag(xPreamble) > 0);
%     I3Preamb = find(real(xPreamble) < 0 & imag(xPreamble) < 0);
%     I4Preamb = find(real(xPreamble) > 0 & imag(xPreamble) < 0);
% 
%     phase1 = mean(angle(rxVec(I1Preamb)))*180/pi;
%     phase2 = mean(angle(rxVec(I2Preamb)))*180/pi;
%     phase3 = mean(angle(rxVec(I3Preamb)))*180/pi;
%     phase4 = mean(angle(rxVec(I4Preamb)))*180/pi;
%     deltaPhase1 = phase1-45;
%     deltaPhase2 = phase2-135;
%     deltaPhase3 = phase3+135;
%     deltaPhase4 = phase4+45;
%     %deltaPhase = (deltaPhase1+deltaPhase2+deltaPhase3+deltaPhase4)/4;
%     rxVec = rxVec*exp(-1i*deltaPhase*pi/180);
%     
%     eucDist = abs(repmat(rxVec.',1,4) - repmat(const, length(rxVec), 1)).^2;
%     [tmp,mHat] = min(eucDist, [], 2);
%     %rxSymbols = const(mHat);
%     rxBitsBuffer = de2bi(mHat'-1, 2, 'left-msb')'; %make symbols into bits
%     rxBits = rxBitsBuffer(:)'; %write as a vector
%    
%     %sum(rxBits(1:22) == preamble)
%     
%     % Step 1: save the estimated bits
%     recObj.UserData.pack = rxBits(23:end);
% 
%     % Step 2: save the sampled symbols
%     recObj.UserData.const = rxVec(12:end)/max(abs(rxVec(12:end)));
% 
%     % Step 3: provide the matched filter output for the eye diagram
%     recObj.UserData.eyed.r = MF_output_conv;
%     recObj.UserData.eyed.fsfd = fsfd;
% 
%     % Step 4: Compute the PSD and save it. 
%     % !!!! NOTE !!!! the PSD should be computed on the BASE BAND signal BEFORE matched filtering
%     [pxx, f] = pwelch(rxBaseband,1024,768,1024, fs); % note that pwr_spect.f will be normalized frequencies
%     f = fftshift(f); %shift to be centered around fs
%     f(1:length(f)/2) = f(1:length(f)/2) - fs; % center to be around zero
%     p = fftshift(10*log10(pxx/max(pxx))); % shift, normalize and convert PSD to dB
%     recObj.UserData.pwr_spect.f = f;
%     recObj.UserData.pwr_spect.p = p;
% 
%     % In order to make the GUI look at the data, we need to set the
%     % receive_complete flag equal to 1:
%     recObj.UserData.receive_complete = 1;    
% 
% %     figure;
% %     subplot(3,1,1)
% %     plot(real(rx))
% %     title('Rx Baseband with no frame sync')
% %     subplot(3,1,2)
% %     plot(real(rxBaseband))
% %     title('Rx baseband after framce sync')
% %     subplot(3,1,3)
% %     plot(real(MF_output_conv))
% %     title('MF output after frame sync')
% % %     
% %     figure;
% %     eyediagram(MF_output_conv, fsfd, 1/Rs);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
if recObj.UserData.counter < 10
    recObj.UserData.counter = recObj.UserData.counter + 1;

else
    stop(recObj);
    disp('Stopped')
    rec_data = getaudiodata(recObj);
    
    %Setup info
    %fs = 44100; %sampling frequency
    %fc = 1000;
    fc = recObj.UserData.fc;
    fs = recObj.UserData.fs;
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
    
    %Root raised cosine pulse
    [pulse, t] = rtrcpuls(alpha,Ts,fs,span);
    
    
    %soundsc(rec_data,fs,24)
    rxBaseband = rec_data'.*exp(-1i*2*pi*fc*(0:length(rec_data')-1)*Tsamp); %down modulate
    rx = rxBaseband;
    
    %Make baseband preamble sequence
    preamble = [1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 1 0 0 1 0];   %preamble to be used -  2x 11 BC
    %preamble = [1 1 1 1 1 1 1 1];
    mPreamble = buffer(preamble, bpsymb)';             %Group 2 bits per symbol (each row will be a symbol)
    mPreIdx = bi2de(mPreamble, 'left-msb')'+1;    % Bits to symbol index, msb: the Most Significant Bit
    xPreamble = const(mPreIdx);                   % Look up symbols using the indices
    xPreUpsample = upsample(xPreamble,fsfd);      % Space the symbols fsfd apart, to enable pulse shaping using conv.
    sPreamble = conv(pulse,xPreUpsample);         %The baseband pulse shaped preamble sequency (ask about this in the Q&A)
    
    %Perform match filter before correlating as a test
    %rxBaseband = matchFilter(pulse,rxBaseband);

    corr = conv((rxBaseband), fliplr(sPreamble));   % correlate the sequence and rx_baseband
    plot(abs(corr))
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %trying to invert the signal
    if abs(min(real(corr))) > abs(max(real(corr)))
        rxBaseband = rxBaseband*exp(1i*pi); %rotate by 180 degrees;
        disp('Yes')
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [tmp, Tmax] = max(abs((corr)));         %find location of max correlation (this should be where the preamble starts)
    Tmax
    %fprintf('delay = %d \n',Tmax-length(sPreamble));
    delay = Tmax - length(sPreamble); %this point will be where the preamble starts, so for rx_baseband we will take from this point onwards?
    if delay < 0
        delay = 0;
    end
    if tmp > 4
        rxBaseband = rxBaseband(delay+3:47798+delay+3); %cut out the useless signal from rx_baseband (also ask if this is correct way)
    
    MF_output_conv = matchFilter(pulse,rxBaseband);
    rxVec = MF_output_conv(1:fsfd:end); 
    rxVec = rxVec(1:227);
    
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
    
    scatterplot(rxVec/max(abs(rxVec))); %scatterplot of received symbols
    
    %length(rxVec)
    
    eucDist = abs(repmat(rxVec.',1,4) - repmat(const, length(rxVec), 1)).^2;
    [tmp mHat] = min(eucDist, [], 2);
    rxSymbols = const(mHat);
    rxBitsBuffer = de2bi(mHat'-1, 2, 'left-msb')'; %make symbols into bits
    rxBits = rxBitsBuffer(:)'; %write as a vector
    


    sum(rxBits(1:22) == preamble)
    %sum(rxBits(393:400) == [0 1 1 0 0 1 0 0]);
    %rxBits(23:30)
    
    figure;
    subplot(3,1,1)
    plot(real(rx))
    title('Rx Baseband with no frame sync')
    subplot(3,1,2)
    plot(real(rxBaseband))
    title('Rx baseband after framce sync')
    subplot(3,1,3)
    plot(real(MF_output_conv))
    title('MF output after frame sync')
    
    %eyediagram(MF_output_conv, fsfd, 1/Rs); % plot the eyediagram from the output of matched filter using MATLAB's function
    
    %figure; plot(abs(corr), '.-r'); title('Correlation between rx_{baseband} and preamble pulse sequence')       % plot correlation
    figure;
    subplot(3,1,1)
    plot(abs(corr))
    title('Abs correlation')
    subplot(3,1,2)
    plot(real(corr))
    title('Real Corr')
    subplot(3,1,3)
    plot(imag(corr))
    title('Imag corr')
    
%     figure;
%     plot(abs(corr))
%     hold on
%     plot(abs(corr1))

    % Step 1: save the estimated bits
    recObj.UserData.pack = rxBits(23:end);

    % Step 2: save the sampled symbols
    recObj.UserData.const = rxVec(12:end)/max(abs(rxVec(12:end)));

    % Step 3: provide the matched filter output for the eye diagram
    recObj.UserData.eyed.r = MF_output_conv;
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
    recObj.UserData.receive_complete = 1;
    end
end
%}
end

function MF_output_conv = matchFilter(pulse, signal)
    MF = fliplr(conj(pulse));        %create matched filter impulse response
    %MF_output = filter(MF,1,signal);      % run received signal through matched filter
    %figure; plot(real(MF_output))
    %MF_output = MF_output(length(MF):end-length(MF)+1); %remove transient
    MF_output_conv = conv(pulse, signal);  % Another approach to MF using conv, what's the difference?
    %figure; plot(real(MF_output))
    MF_output_conv = MF_output_conv(length(MF):end-length(MF)+1);
    %rxVec = MF_output_conv(1:fsfd:end);  %get sample points
end

function [preambleTrain,xPreamble] = makePreamble(pulse,samplesSymb,bitsSymb)
    preamble = kron([1 1 0 1],[1 1 1 1 1 0 0 1 1 0 1 0 1]);
    %preamble = [1 1 1 1 1 0 0 1 1 0 1 0 1];
    %preamble = [preamble preamble];
    const = [(1+1i), (1-1i), (-1-1i), (-1+1i)]/sqrt(2);     %qpsk constelation
    mPreamble = buffer(preamble, bitsSymb)';            %Group 2 bits per symbol (each row will be a symbol)
    mPreIdx = bi2de(mPreamble, 'left-msb')'+1;              % Bits to symbol index, msb: the Most Significant Bit
    xPreamble = const(mPreIdx);                             % Look up symbols using the indices
    xPreUpsample = upsample(xPreamble,samplesSymb);         % Space the symbols fsfd apart, to enable pulse shaping using conv.
    preambleTrain = conv(pulse,xPreUpsample);               %The baseband pulse shaped preamble sequency (ask about this in the Q&A)
end
