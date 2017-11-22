function [] = saveEEGdata(sujeto) %edf to mat
%filename = nombre con extensión. Guarda edf a mat con mismo nombre
%data es timestamps eeg_data gyro_data
path = pwd; %/Users/antonialarranaga/Desktop/Memoria - scripts
fileEdf = strcat(path, '/sujetos/', sujeto, '/',sujeto, '.edf');
[ nSeqUnixEEG, channels_labels, channels_data, gyro_labels, gyro_data ] = getEEGdata(fileEdf);
channels_data = cell2mat(channels_data);
gyro_data = cell2mat(gyro_data);
data = [nSeqUnixEEG channels_data gyro_data];
filemat = strcat(path, '/sujetos/', sujeto, '/', sujeto, '_EEG.mat');
save(filemat, 'data')