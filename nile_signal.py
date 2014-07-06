%pylab inline

#Import data from a text file
fp = open("FloodsOfTheNile_data.txt", "rb")
FN = loadtxt(fp ,delimiter=",",skiprows=2)
fp.close()

#Linearize the matrix
amplitude = reshape(FN[:,1:-1], (-1,1))

#create time index, slices of 1/12 are for months
time = arange(FN[0,0],FN[-1,0]+1,1./12.)

#identify unavailable values
I = np.nonzero(amplitude >= 0)[0]

#identify start of continuous values, start in january
s = (ceil(max(nonzero(amplitude < 0)[0])/12)+1)*12

#compact all this
T = time[s:]
A = array([float(amplitude[i]) for i in range(int(s),len(amplitude))])

figure()
plot(time[I], amplitude[I])
title('Floods of the Nile')
xlim(time[0], time[-1])
xlabel('Year')
ylabel('Flow [m$^3$/s]')
show()

figure()
plot(range(1,13), reshape(amplitude[s:], (-1,12)).T)
title('Yearly curves overlapped')
xlim((0.95, 12.05))
xticks(arange(1,13), ('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'), rotation='vertical')
xlabel('Month')
ylabel('Flow [m$^3$/s]')
show()

figure()
boxplot(reshape(amplitude[s:], (-1,12)),1)
title('Yearly curves overlapped')
xlim((0.95, 12.05))
xticks(arange(1,13), ('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'), rotation='vertical')
show()

#longterm dependence

flood = A[8::12]
flood = [float(flood[i]) for i in range(len(flood))]
t = T[1::12]

plot(t, flood)
xlabel('Year')
ylabel('Flow [m$^3$/s]')
show()

#autocorrelation

def acf(x):
    R = zeros(size(x))
    mu = mean(x)
    sigma2 = std(x)**2
    n = len(x)
    for k in range(n):
        R[k] = sum((x[0:n-k]-mu)*(x[k:]-mu))/(sigma2*(n-k))
        
    return R

t = arange(0,len(flood)/2)
acf_flood = acf(flood)[:len(t)]
acf_rand = acf(rand(size(flood)))[:len(t)]

plot(t, acf_flood[:len(t)], 'b')
plot(t, acf_rand[:len(t)], 'r')
legend(('Flood data', 'Random'))
show()

#hurst exponent

#Compute the rescaled range for all partial time series of length n
def rescaled_range(X,n):   
    
    RS = 0
    
    for i in range(int(len(X)-n)):
        
        #compute the mean
        m = mean(X[i:i+n])
        # make zero mean
        Y = X[i:i+n] - m
        #partial cumulative sums
        Z = cumsum(Y)
        #difference max and min of this time-serie
        R = max(Z) - min(Z)
        #variance
        S = sqrt(mean(Y**2))
        
        RS += R/S
    
    return RS/(len(X)-n)
    
#Routine to compute hurst factor
def hurst_factor(data):
    
    #compute rescaled range for all size of partial sums
    logmax = int(floor(log2(len(data))))
    N = 2**arange(2,logmax+1)
    RS = zeros(size(N))
    for i in arange(logmax-1):
        RS[i] = rescaled_range(data, N[i])
    
    #compute the linear fit
    M = reshape(concatenate((log(N.T), ones(N.shape).T), axis=0), (2,-1)).T
    lf = dot(dot(inv(dot(M.T, M)), M.T), log(RS).T)
    
    #Hurst parameter is the slope of the linear fit
    return lf[0], lf, log(N), RS


#Compute Hurst factor
hf, lf, n, RS = hurst_factor(flood)

#plot empirical rescaled range and linear fit
plot(n, log(RS), 'b', n, lf[1]+n*lf[0], 'r--')
xlim(n[0],n[-1])
xlabel('log(n)')
ylabel('log(R[n]/S[n])')
show()

print 'Hurst factor for the Floods of the Nile : %f' % lf[0]

#check if follows decay of the ACF

alpha = 2*(1-hf)
plot(t, acf_flood)
acf_decay = arange(0,len(acf_flood))**(-alpha)
plot(t, acf_decay,'r')
show()

#check for random uncorrelated data

hfr, lfr, nr, RSr = hurst_factor(randn(1000))

# plot empirical rescaled range

plot(nr, log(RSr),'b', nr, lfr[0]*nr+lfr[1], 'r--')
xlim((nr[0],nr[-1]))
ylabel('log(R/S)')
show()

print 'Hurst factor for random data : %f' % lfr[0]

#discrete Fourier transform

N = len(A)
Ts = 1./12.
Ttot = N*Ts

#Compute the DFT
F = fft.fft(A.T).T

#since the signal is real, we can just keep one half of the spectrum
F = F[:N/2+1]

#construct the corresponding frequency vector
freq = arange(0, N/2+1)/Ttot

#display the magnitude of the spectrum
figure()
plot(freq, np.abs(F))
xlabel('1/year')
ylabel('|dft(A)|')
xlim((-0.1,6.1))
show()

zero_range = range(0, len(F), int(Ttot))
new_F = F
new_F[zero_range] = 0
figure()
plot(freq, abs(new_F))
xlabel('1/year')
ylabel('|dft(A)|')
xlim((-0.1,6.1))
show()