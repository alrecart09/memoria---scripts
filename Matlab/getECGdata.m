
function [ timeECG, dataECG  ] = getECGdata( filename )
%% Open h5 file - openSignals (ECG)
%% read file and info
hinfo = hdf5info(filename);
%% read attribute
attributename = hinfo.GroupHierarchy.Groups.Attributes;
samplingRate = double(attributename(1, 3).Value);
date = attributename(1, 13).Value.Data;
timeStart = attributename(1, 14).Value.Data;
mac = hinfo.GroupHierarchy.Groups.Name;
%duration = attributename(1, 16).Value.Data;

%% read raw data
nSeq = double(hdf5read(filename,[mac,'/raw/nSeq']));
ch3 = double(hdf5read(filename,[mac, '/raw/channel_3'])); %ECG [v]
%ch4 = hdf5read(filename,'/98:D3:31:80:48:1B/raw/channel_4'); %Acc

%% timeStart (local time) to Unix timeStamp 
timeStartUnix = localTime2unixTimeECG( date, timeStart ); %[ms]

%% Sync 
period = (1000/samplingRate); % [ms] 
aux =  0:length(nSeq)-1;
nSeqUnixECG = timeStartUnix+(aux*period); %[ms]

%% Convert ECG values to real units [-1.5 mV; 1.5 mV]
ECG_signal = ch3;
Vcc = 3.3;
Gecg = 1100;  
n =10;  %Number of Bits
ECG_v = (( (ECG_signal.*Vcc )/ (2^n - (Vcc/2) ) ) / Gecg);
ECG_mv = ECG_v*1000; 

%%
dataECG = ECG_mv';
timeECG = nSeqUnixECG';
%%
%---------value 'n'-------%%
%- If it is one of the FOUR INITIAL channels, its resolution will be: 10 bit. 
%- If it is one of the two last channels, its resolution will be: 6 bit.

%%
clear filename hinfo samplingRate attributename timeStart nSeq ch3 ...
    date timeStartUnix period ECG_signal Vcc Gecg n ECG_v;
 
end