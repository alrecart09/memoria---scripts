function [] = saveECGdata(sujeto)
%name = nombre usuario (carpeta del archivo)
%data es info ecg, acc y timestamps
path = pwd; %/Users/antonialarranaga/Desktop/Memoria - scripts
fileH5 = strcat(path, '/sujetos/', sujeto, '/', sujeto, '.h5');
[ecgtime, ecgdata] = getECGdata(fileH5);
data = [ecgtime ecgdata];
filemat = strcat(path, '/sujetos/', sujeto, '/', sujeto, '_ECG.mat');
save(filemat, 'data')
end

