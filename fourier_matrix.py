import scipy.io

#make DFT matrix
def dftmatrix(N):
	import numpy as np

		#create 1xN matrix with indices from 0 to N-1

		a = np.expand_dims(np.arange(N), 0)

		#create matrix

		WN = np.exp(-2j*np.pi*a.T*a/N)

		return WN

mat = scipy.io.loadmat('frequencyRepresentation.mat')
print mat

signal = mat['x']
print signal
type(signal)

#visualize signal

signal = signal.reshape(signal.size,)
plot(np.arange(signal.size),signal)
xlim([0,500])
xlabel('Samples')
ylabel('x')
show()

#transformation matrix

signal = signal.reshape(signal.size,1)
W4000 = dftmatrix(4000)
X = np.dot(W4000,signal)

normFrequ = np.arange(1,X.size+1,dtype=float)/float(X.size)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()

#complex sinusoids

x = np.append(np.ones(64),np.zeros(64))
W128 = dftmatrix(128)
X1 = np.dot(W128[0],x)
xx1 = np.real(np.conjugate(W128[:,0])*X1*1/128)

X2 = np.dot(W128[0:2],x)
xx2 = np.real(np.dot(np.conjugate(W128[:,0:2]),X2)*1/128)

X3 = np.dot(W128[0:3],x)
xx3 = np.real(np.dot(np.conjugate(W128[:,0:3]),X3)*1/128)

X4 = np.dot(W128[0:4],x)
xx4 = np.real(np.dot(np.conjugate(W128[:,0:4]),X4)*1/128)

X5 = np.dot(W128[0:5],x)
xx5 = np.real(np.dot(np.conjugate(W128[:,0:5]),X5)*1/128)

X6 = np.dot(W128[0:6],x)
xx6 = np.real(np.dot(np.conjugate(W128[:,0:6]),X6)*1/128)

X7 = np.dot(W128[0:7],x)
xx7 = np.real(np.dot(np.conjugate(W128[:,0:7]),X7)*1/128)

X8 = np.dot(W128[0:8],x)
xx8 = np.real(np.dot(np.conjugate(W128[:,0:8]),X8)*1/128)

X9 = np.dot(W128[0:9],x)
xx9 = np.real(np.dot(np.conjugate(W128[:,0:9]),X9)*1/128)

X10 = np.dot(W128[0:10],x)
xx10 = np.real(np.dot(np.conjugate(W128[:,0:10]),X10)*1/128)

plot(np.arange(xx1.size),xx1)
plot(np.arange(xx2.size),xx2)
plot(np.arange(xx3.size),xx3)
plot(np.arange(xx4.size),xx4)
plot(np.arange(xx5.size),xx5)
plot(np.arange(xx6.size),xx6)
plot(np.arange(xx7.size),xx7)
plot(np.arange(xx8.size),xx8)
plot(np.arange(xx9.size),xx9)
plot(np.arange(xx10.size),xx10)
xlabel('Samples')
ylabel('x')
xlim([0,128])
show()

