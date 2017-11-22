function [data, ppg] = gsr2mat(name) 
%name = nombre carpeta usuario
path = pwd; %/Users/antonialarranaga/Desktop/Memoria - scripts
fileGSRPPG = strcat(path, '/sujetos/', name, '/', name, '_GSR_PPG.csv');
data = csvread(fileGSRPPG, 3, 0);
timestamps = data(:, 1);
gsr_ = data(:, 2);
ppg_ = data(:, 3);

events = struct();
events.time = 0;
events.nid = 0;
events.name = '0';
events.userdata = [];

data = struct('conductance', gsr_', 'time', timestamps', 'timeoff', 0, 'event', events);
%gsr = [timestamps gsr_];
ppg = [timestamps ppg_];

filemat_gsr = strcat(path, '/sujetos/', name, '/', name, '_GSR.mat');
filemat_ppg = strcat(path, '/sujetos/', name, '/', name, '_PPG.mat');

save(filemat_gsr, 'data')
save(filemat_ppg, 'ppg')

end

