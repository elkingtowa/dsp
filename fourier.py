#discrete Fourier transform (DFT) of a signa

#make DFT matrix
def dftmatrix(N):
	import numpy as np

		#create 1xN matrix with indices from 0 to N-1

		a = np.expand_dims(np.arange(N), 0)

		#create matrix

		WN = np.exp(-2j*np.pi*a.T*a/N)

		return WN

#multiply 3x3 matrix (WN) and input signal (X)

N = 6
WN = dftmatrix(N)

x = np.ones(N)
X = np.dot(WN,x)

#calculate inverse transformation

x2 = 1./N * np.dot(w)(WN.T.conjugate(), X)
print 'Transformation Error'
print (np.abs(x-x2).sum())

