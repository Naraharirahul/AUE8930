%% 1 a)
clc
clear all
Fs = 1000;
T = 1/Fs;
L = 1000;
t = (0:L-1)*T;
x =  2  + 3*cos(500*pi*t) + 2* cos(1000*pi*t) + 3*sin(2000 * pi * t);
figure(1)
plot(t,x);
title('Continous period signal')
xlabel('time t')
ylabel('Amplitude')

% 1 b)
Y = fft(x); 
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
figure(2)
plot(f,P1) 
title('Fast Fourier transform')
xlabel('f (Hz)')
ylabel('Amplitude')


%%
% 2a
clc
clear all
for i = 1:999
    if i > 499
        x(i) = 1;
    else
        x(i) = 0;
    end     
end

t = 1:999;
figure(1)
plot(t,x)
title('Discrete period signal')
xlabel('time t')
ylabel('Amplitude')

%2b
Fs = 1000;
L = 1500;
Y = fft(x); 
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
figure(2)
plot(f,P1) 
title('Fast Fourier transform')
xlabel('f (Hz)')
ylabel('Amplitude')
half_power = powerbw(Y);
high_freq = max(P1);
low_freq = min(P1);

%%
%3a
clc
clear all
for i = 1:499
    x(i) = 20;
end
t = 1:499;
x_n = x + randn(size(t));

figure(1)
plot(t,x)
hold on
plot(t,x_n)
xlabel('Time');
ylabel('Amplitude');
title('normal distributed noise added to a signal');
hold off

gk3 = [0.27901,0.44198,0.27901];
m=length(x_n);
n=length(gk3);
X=[x_n,zeros(1,n)];
H=[gk3,zeros(1,m)];
for i=1:n+m-1
    Y(i)=0;
    for k=1:m
        if(i-k+1>0)
            Y(i)= Y(i) + X(k)*H(i-k+1);
        end
    end
end
figure(2);
plot(t,x_n);
hold on
plot(Y);
xlabel('Time');
ylabel('Amplitude');
title('gaussian kernel window size 3');
hold off

gk11 = [0.000003 0.000229 0.005977 0.060598 0.24173	0.382925 0.24173 0.060598 0.005977 0.000229 0.000003];
m=length(x_n);
n=length(gk11);
X=[x_n,zeros(1,n)];
H=[gk11,zeros(1,m)];
for i=1:n+m-1
    Y1(i)=0;
    for k=1:m
        if(i-k+1>0)
            Y1(i)= Y1(i) + X(k)*H(i-k+1);
        end
    end
end
figure(3);
plot(t,x_n);
hold on
plot(Y1);
xlabel('Time');
ylabel('Amplitude');
title('gaussian kernel window size 11');
hold off
