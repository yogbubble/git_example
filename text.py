import numpy as np
import math
import os.path
import matplotlib.pyplot as plt
import pylab as pl
from scipy.optimize import curve_fit
from scipy.stats import linregress
from numpy.polynomial import polynomial
from scipy.optimize import leastsq
import scipy.fftpack as fftpack
import scipy.optimize as optimize
from scipy.fftpack import fft

//the reading file function
x, z, y = np.loadtxt('LAM.txt', unpack=True)
x1, z1, y1 = np.loadtxt('HEX.txt', unpack=True)

z1 = np.array(np.float32(z1))
z =np.array(np.float32(z))

cross=[]
crossy=[]
crossx=[]


testx1 = []
testF1 = []
testx2 = []
testF2 = []

for i in range(len(z)):
	if(abs(z1[i]-z[i])<=0.003):
		cross.append(z1[i])
		crossy.append(y1[i])
		crossx.append(x1[i])

Width = []
ratio = []

fraction = []
Freeenergy = []

for i in range(0,91):
	Width.append(17.0+0.2*i)

for i in range(0,100):
	ratio.append(0.3+0.002*i)

Width =np.array(Width)	
ratio =np.array(ratio)	


def f_fit(x, a, b, c):
   return a*x*x + b*x + c

for m in range(len(Width)):
	for u in range(0,len(x)):
		if (y[u]==Width[m]):
			testF1.append(z[u])
			testF2.append(z1[u])
			testx1.append(x[u])
			testx2.append(x1[u])
		else:
			continue
	
	testx2 = np.array(testx2)
	testF1 = np.array(testF1)		
	testx1 = np.array(testx1)		
	testF2 = np.array(testF2)		
			

	if (len(testx1)!=0):
		par1,cov1=optimize.curve_fit(f_fit,testx1,testF1,maxfev=10000)
	 	par2,cov2=optimize.curve_fit(f_fit,testx2,testF2,maxfev=10000)		

	 	solution1= ((par2[1]-par1[1])+np.sqrt((par2[1]-par1[1])**2 - 4*(par2[0]-par1[0])*(par2[2]-par1[2])))/(2*(par1[0]-par2[0]))
	 	solution2= ((par2[1]-par1[1])-np.sqrt((par2[1]-par1[1])**2 - 4*(par2[0]-par1[0])*(par2[2]-par1[2])))/(2*(par1[0]-par2[0]))

	fraction.append(solution2)
	Freeenergy.append(Width[m])

	plt.figure()
	# plt.plot(fraction,Freeenergy, c="g", lw=3.5)
	plt.scatter(testx1,testF1, c="b", alpha=0.5,label='LAM')
	plt.plot(testx1,f_fit(testx1,par1[0],par1[1],par1[2]),color='c',lw=6,label=r'$LAM \ fit$')
	plt.scatter(testx2,testF2, c="r", alpha=0.5,label='HEX')
	plt.plot(testx1,f_fit(testx1,par2[0],par2[1],par2[2]),color='brown',lw=6,label=r'$HEX \ fit$')
	plt.legend()


	plt.xlabel("f")
	plt.ylabel(r"$Free \ Energy$")
	plt.show()

	


	testx1 = []
	testF1 = []
	testx2 = []
	testF2 = []

plt.figure()
plt.plot(fraction,Freeenergy, c="g", lw=3.5)
plt.xlabel(r"$f_A$")
plt.ylabel(r"$\chi N$")
plt.show()





















