%
% perform SSDRC to an audio file.
%
% Aki Kunikoshi
% aki.kunikoshi@readspeaker.com
%


wavDir_in  = '/home/akikun/Corpus/hikari_tts/wavs_normalized';
wavDir_out = '/home/akikun/Corpus/hikari_tts/wavs_ssdrc';
wavPaths = dir(wavDir_in);

for i = 1:length(wavPaths)
    wavName = wavPaths(i).name;
    if length(wavName) > 2
        disp(wavName);
        wavPath_in  = strcat(wavDir_in, '/', wavName);
        wavPath_out = strcat(wavDir_out, '/', wavName); 
        [y, fs] = audioread(wavPath_in);
        y_ssdrc = SSDRC(y, fs);
        audiowrite(wavPath_out, y_ssdrc, fs);
    end
end
clear i wavName wavPath_in wavPath_out
clear y fs 
