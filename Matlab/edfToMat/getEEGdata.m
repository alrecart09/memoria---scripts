%% Open edf file - Emotiv Xavier TestBench (EEG)
%% read file
function [ nSeqUnixEEG, channels_labels, channels_data, gyro_labels, gyro_data ] = getEEGdata( filename )
[dataEEG, headerEEG] = readEDF(filename);

%% read attribute
samplingRate = headerEEG.samplerate(1);
date = strcat(headerEEG.startdate,'17'); %OJO ACA - pq stardate es hasta 20??
timeStart = headerEEG.starttime;

%% get raw data
channels_labels = headerEEG.labels(3:16);
channels_data = dataEEG (3:16);  %uV

gyro_labels = headerEEG.labels(18:19);
gyro_data = dataEEG(18:19);

%% generate time vector
nSeq = linspace(0,length(dataEEG{1,1}),length(dataEEG{1,1})+1);
nSeq = nSeq(1:end-1)';

%% timeStart (local time) to Unix timeStamp (falta automatizar esto)
timeStartUnix = localTime2unixTimeEEG( date, timeStart ); %%1463142069494; %[ms]


%% Sync 
period = (1000/samplingRate); % [ms] 
nSeqUnixEEG = timeStartUnix+(nSeq*period); %[ms]


%%
clear filename samplingRate timeStart nSeq ch3 ...
    date period timeStartUnix ;
 
end