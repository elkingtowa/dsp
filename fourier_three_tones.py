from scipy import constants as c

#make DFT matrix
def dftmatrix(N):
	import numpy as np

		#create 1xN matrix with indices from 0 to N-1

		a = np.expand_dims(np.arange(N), 0)

		#create matrix

		WN = np.exp(-2j*np.pi*a.T*a/N)

		return WN

y = np.linspace(0,999,1000)
x1 = np.sin((y*2*c.pi*40)/1000)
x2 = np.sin((y*2*c.pi*80)/1000)
x3 = np.sin((y*2*c.pi*160)/1000)

#N=5000 signals
x = np.append(x1,[np.zeros(1000),x2,np.zeros(1000),x3])

#Fourier matrix
W5000 = dftmatrix(5000)
X = np.dot(W5000,x)

#magnitude of Fourier transformation
normFrequ = np.arange(1,X.size+1,dtype=float)/float(X.size)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()

W5000reduced = np.append(W5000[0:600],W5000[4399:5000],axis=0)
iW5000reduced = np.conj(np.append(W5000[:,0:600],W5000[:,4399:5000],axis=-1))

#DFT/DFS on a reduced base
Xapprox = np.dot(W5000reduced,x)
#inverse DFT/DFS of the DFT/DFS on a reduced base
xapprox = np.dot(iW5000reduced,Xapprox)*1/5000
#real value of the result to get rid of imaginary numerical errors
xapprox = np.real(xapprox)

X = np.dot(W5000,xapprox)

plot(np.arange(1,5001,dtype=float)/float(5000),abs(X))
show()