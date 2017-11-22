function [] = ledaLabBatch(name)
%Sobreescribe archivo _GSRLedalab.mat post preprocesameinto y
%analisis CDA
% No abrir Ledalab pq sino se cae pq no se cierra(?)
path = '/Users/antonialarranaga/Desktop/Memoria - scripts'; %/Users/antonialarranaga/Desktop/Memoria - scripts

data = strcat(path, '/sujetos/', name, '/GSR/', name, '_GSRLedalab.mat');
%name = nombre carpeta usuario

if strcmp(name,'hector-otarola') || strcmp(name, 'tomas-lagos')
    Ledalab(data, 'open','mat', 'downsample', 0, 'smooth',{'hann',15}, 'analyze','CDA', 'optimize',6, 'overview',1)

else
    Ledalab(data, 'open','mat', 'filter', [1 5], 'downsample', 12, 'smooth',{'hann',15}, 'analyze','CDA', 'optimize',6, 'overview',1)
end

end

