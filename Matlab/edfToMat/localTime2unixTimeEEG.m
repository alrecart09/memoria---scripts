function [ unixTime ] = localTime2unixTimeEEG( date, time )
%Transforma la hora local en formato Unix en milisegundos
% 
timezone = 3; % UTC 00 esta 3 horas antes q nosotros OJO CON CAMBIOS DE HORA!!
days = datenum(date,'dd.mm.yyyy') - 719529; %719529 = days from 1-1-0000 to 1-1-1970
date_ms = days*86400000; % 86400000 miliseconds in a day

%%
h_ms = (str2double(time(1:2))+timezone)*3600000;
m_ms = str2double(time(4:5))*60000;
ms = str2double(time(7:8))*1000;

%%
unixTime = date_ms + h_ms + m_ms + ms; 
end

