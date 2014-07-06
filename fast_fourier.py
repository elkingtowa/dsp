from scipy import fftpack as f

#reduce number of multiplications to 2Nlog2(N)

#make DFT matrix
def dftmatrix(N):
	import numpy as np

		#create 1xN matrix with indices from 0 to N-1

		a = np.expand_dims(np.arange(N), 0)

		#create matrix

		WN = np.exp(-2j*np.pi*a.T*a/N)

		return WN

#two tones signal

y = np.linspace(0,3999,4000)
x1 = np.sin((y*2*c.pi*40)/1000)
x2 = np.sin((y*2*c.pi*50)/1000)
x = x1 + x2

M = 4000
X = f.fft(x,M)
#Plot versus the normalized frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()

#Plot versus the normalized frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
ylim([0,35])
show()

#inverse
x4000 = f.ifft(X)

#zero pad signal

X4000 = f.fft(x50,M)

#Plot versus the normalized Frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X4000))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()
