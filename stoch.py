import math
import curses
import random
import sys
import numpy as np
import scipy as scpy
from scipy import linalg
import matplotlib.pyplot as plt
import threading
import time
from bigfloat import *
from fractions import Fraction


HEADER = '\033[95m'
OKBLUE = '\033[44m'
YELLOWUL =  '\033[93m \033[4m '
OKGREEN = '\033[92m'
WARNING = '\033[91m'
BGRED = '\033[101m'
BGGREEN = '\033[102m' 
FAIL = '\033[91m'
ENDC = '\033[0m'

fptr = None
logpriority = 1
prtval = 0

def writelog(priority, msg):
	if (priority <= logpriority):
		threading.current_thread().fptr.write(msg)
		threading.current_thread().fptr.flush()


class output:
	def __init__(self):
		self.mdp = None
		self.vu = None
		self.vJ = None
		self.pu = None
		self.pJ = None
		self.estsd = None
		self.std = None 
		self.vstJv = None
		self.vstTDJv = None
		self.vstMCJ = None
		self.pvJ = None
		self.tds = None
		self.lstds = None
		self.rlstds = None
		self.gtd2s = None
		self.gtd2v2s = None
		self.lstd0 = None
		self.lspes = None
		self.bert1 = None
		self.bert2 = None
		self.bert3 = None
		self.ftheta = None
		self.lsce = None
	
	def store1(self, mdp, estsd, std, vu, vJ, pu, pJ, vstJv, vstTDJv, vstMCJ):
		self.mdp = mdp
		self.estsd = estsd
		self.std =std 
		self.vu = vu
		self.vJ = vJ
		self.pu = pu 
		self.pJ = pJ 
		self.vstJv = vstJv 
		self.vstTDJv = vstTDJv
		self.vstMCJ = vstMCJ

	def store2(self, mdpobj, std, vu, vJ, pvJ, tds, lstds, rlstds, gtd2s, gtd2v2s, lstd0, lspes, bert1, bert2, bert3, ftheta, lsce):
		self.mdp = mdpobj
		self.std = std
		self.vu = vu
		self.vJ = vJ
		self.pvJ = pvJ 
		self.tds = tds
		self.lstds = lstds
		self.rlstds = rlstds
		self.gtd2s = gtd2s
		self.gtd2v2s = gtd2v2s
		self.lstd0 = lstd0
		self.lspes = lspes
		self.bert1 = bert1
		self.bert2 = bert2
		self.bert3 = bert3
		self.ftheta = ftheta
		self.lsce = lsce 


	def evalBellmanError(self, r):
		global prtval
		prtval = 0
		fval = 0.0
		u = self.vu
		theta = self.ftheta
		J = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		TJ = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		for s in range(self.mdp.ssz):
			op1 = scpy.dot(self.mdp.evalftval(s, theta.flatten()), r) 
			op2 = 0.0
			for t in range(self.mdp.ssz):
				op2 += (self.mdp.P[s][u[s]][t]*(self.mdp.g[s][u[s]][t] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(t, theta.flatten()))))
			TJ[s] = op2
			J[s] = op1

		pTJ = self.mdp.projection(TJ, self.std)
		fval = scpy.dot(self.std[0:np.size(self.std)-1], ((J-pTJ)**2.0))
		return fval



	def printvals(self):
		print OKBLUE, "MDP:", self.mdp.ssz, " states ", self.mdp.asz, " actions ", ENDC
		print "[ Stationary dist ] "+WARNING, self.std[0:len(self.std)-1], ENDC
		#print "[ Estimated stationary dist ] ",self.estsd
		print OKBLUE, '-'*175, ENDC
		print "[ ValueIteration ]"
		print "u*:" , self.vu
		print "J*: ", self.vJ
		print "Proje: "
		print self.mdp.projection(self.vJ, self.std)
		if (False):
			print "[ PolicyIteration ]"
			print "u*: ", self.pu
			print "J*: ", self.pJ
			print OKBLUE, '-'*175, ENDC
			print "TD("u"\u03BB)(J*):"
			printc(self.vstJv, self.vJ)
			print OKBLUE, '-'*175, ENDC
			print "TD(0)(J*): " 
			printc(self.vstTDJv, self.vJ)
			print OKBLUE, '-'*175, ENDC
			print "Monte Carlo(J*): "
			printc(self.vstMCJ, self.vJ)
		err1 = math.sqrt(self.mdp.norm(self.vJ-self.pvJ, self.std))
		print str(self.bert1) , " \n"
		print str(self.bert2) , " \n"
		print str(self.bert3) , " \n"
		err1_2 = self.evalBellmanError(self.bert1)
		err1_3 = self.evalBellmanError(self.bert3)
		err1_4 = self.evalBellmanError(self.bert2)
		err2 = math.sqrt(self.mdp.norm(self.vJ-self.tds, self.std))
		err3 = math.sqrt(self.mdp.norm(self.vJ-self.lstds, self.std))
		err4 = math.sqrt(self.mdp.norm(self.vJ-self.rlstds, self.std))
		err5 = math.sqrt(self.mdp.norm(self.vJ-self.lspes, self.std))
		err6 = math.sqrt(self.mdp.norm(self.vJ-self.lstd0, self.std))
		err7 = math.sqrt(self.mdp.norm(self.vJ-self.gtd2s, self.std))
		err8 = math.sqrt(self.mdp.norm(self.vJ-self.gtd2v2s, self.std))
		err9 = math.sqrt(self.mdp.norm(self.vJ-self.lsce, self.std))
		print OKBLUE, "Function approximation with ", self.mdp.kdim, " RBF", ENDC
		print YELLOWUL, u"\u03C0J*: ", ENDC, self.pvJ, OKGREEN, "@", err1, ENDC
		print YELLOWUL, "TD("u"\u03BB):", ENDC, self.tds, OKGREEN, "@", err2, ENDC
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL, "LSTD("u"\u03BB):", ENDC, self.lstds, OKGREEN, "@", err3, ENDC
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL, "RLSTD(0)  :  ", ENDC, self.rlstds, OKGREEN, "@", err4, "/", err1_4, ENDC
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL, "GTD2:      ", ENDC, self.gtd2s, OKGREEN, "@", err7, ENDC
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL, "GTD2V2:    ", ENDC, self.gtd2v2s, OKGREEN, "@", err8, ENDC
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL, "LSTD(0):     ", ENDC, self.lstd0, OKGREEN, "@", err6, "/", err1_3, ENDC 
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL,"LSPE("u"\u03BB):     ", ENDC, self.lspes, OKGREEN, "@", err5, ENDC
		print OKBLUE, '-'*175, ENDC
		print YELLOWUL,"CE:     ", ENDC, self.lsce, OKGREEN, "@", err9, "/", err1_2,  ENDC
		print OKBLUE, '-'*175, ENDC

class myrand:
	def __init__(self):
		self.rkey = 0

	def getRandom(self, pmf, sz):
		ln = len(pmf)
		sf = np.array([0.0]*(ln+1))
		for i in range(1,ln):
			sf[i] = sf[i-1]+pmf[i-1]*100.0
		sf[ln] = 100.0
		rv = np.array([0]*sz)
		for j in range(sz):
			v = np.random.uniform(0,100)
			for i in range(ln+1):
				if sf[i] >= v:
					break
			rv[j] = i-1	
		return rv
		
class mrasd:
	def __init__(self):
		self.lbda = 1.0/8.0
		self.ITERCOUNT = 100
		self.dlta = 1.5
		self.eps = 0.01
		self.randv = myrand()
		self.theta0 = None
		self.theta0_shr = 1.0

	def getSamples(self, z, sz):
		#return np.random.poisson(p[0], sz)
		#v = z[1]-(z[0]**2.0)
		m = z[0]
		v = z[1]
		print (m, v), sz
		return np.random.normal(m, sqrt(v), sz)


	def testfunc(self):
		return self.evalfuncH(10)

	def getSamples2(self, z, sz, a):
		n = int(z[0])
		p = z[1]
		pmf = [0.0]*n
		for i in range(n):
			pmf[i] = ((1-a)*sc.misc.factorial(n)*(p**float(i))*((1-p)**(float(n-i))))/(sc.misc.factorial(n-i)*sc.misc.factorial(i))
			pmf[i] += (a*1.0)/100.0
		return self.randv.getRandom(pmf, sz)

		
		
	def evalfuncF(self, x, thetak):
		(m, v) = thetak
		return (1.0/scpy.sqrt(2.0*np.pi*v))*scpy.exp(-(x-m)**2/(v*2.0))

	def evalfuncH1(self, x):
		if x > -39.0/6.0 and x <= 39.0/6.0:
			return x*np.cos(3*np.pi*x)
		elif x > 39.0/6.0 and x < 10:
			return (39.0/6.0)*np.cos(3*np.pi*x)/np.sqrt(x);
		elif x > -10 and x <= -39.0/6.0:
			return (-39.0/6.0)*np.cos(3*np.pi*x)/np.sqrt(np.fabs(x))
		else:
			return 0

	def evalfuncH(self, x):
		if x in range(0, 30):
			return (x*np.cos(3.0*np.pi*x))
		elif x  in range(30, 60):
			return ((x-20.0)*np.cos(3*np.pi*x))
		elif x in range(60, 90):
			return ((x-40.0)*np.cos(3*np.pi*x))
		elif x in range(90, 100):
			return (10.0*np.cos(3*np.pi*x))
		else:
			return 0

	def getFuncVals(self, X):
		V = np.array([0.0]*len(X))
		for i in range(len(X)):
			V[i] = self.evalfuncH(X[i])
		return V
		

	def getQuantile(self, V, p):
		sV = np.sort(V)
		idx = int(math.floor(p*len(V)))
		if idx > 0:
			rv = sV[idx-1]
		else:
			rv = sV[idx]
		return rv


	def evalfuncS(self, x):
		r = 0.1
		return scpy.exp(r*x, dtype=np.longdouble)

	def evalfunc101(self, x, hx, g, thetak, k):
		fv = 0.0
		if hx >= g:
			cst = (self.evalfuncS(hx)**k)#/self.evalfuncF(x, thetak)
			fv = cst*x
			#fv = x*self.evalfuncS(self.evalfuncH(x))**k

		return fv


	def evalfunc102(self, hx, g, thetak, k):
		fv = 0.0
		if hx >= g:
			fv = self.evalfuncS(hx)**k#/self.evalfuncF(x, thetak)
			#fv = self.evalfuncS(self.evalfuncH(x))**k

		return fv

	def evalfunc103(self, x, hx, g, thetak, k, m):
		fv = scpy.array([[0.0]*len(m)]*len(m), dtype=np.longdouble)
		if hx >= g:
			cst = self.evalfuncS(hx)**k#/self.evalfuncF(x, thetak)
			op1 = (x-m)
			op2 = (x-m)
			op1.resize(len(op1),1)
			op2.resize(1,len(op2))
			rs = scpy.dot(op1,op2)
			fv = cst*rs
			#fv = (x-m)**2*self.evalfuncS(self.evalfuncH(x))**k

		return fv


	def computeTheta(self, thetak, N, X, V, g, k):
		(t1, t2, t3) = (0.0, 0.0, 0.0)
		try:
			for i in range(1, N+1):
				t1 += self.evalfunc101(X[i-1], V[i-1], g, thetak, k)
				t2 += self.evalfunc102(V[i-1], g, thetak, k)
			nm = t1/t2
			for i in range(1, N+1):
				t3 += self.evalfunc103(X[i-1], V[i-1], g, thetak, k, nm)
			nv = t3/t2
		except IndexError:
			print "IndexError: ", i-1, Nk
			exit(1)

		return (nm, nv)
		
	def evalfuncphi(self, x):
		r = 0.01
		return scpy.exp(r*x, dtype=np.longdouble)

	def evalfuncI(self, x, g):
		ep = 0.01
		if x >= g:
			return 1
		#elif x > g-ep:
		#	return (x-g+ep)/ep
		else:
			return 0
		

	def evalfuncTheta(self, x):
		return np.array([x, x*x])


	def evalfunc1(self, X, g):
		q = np.array([0.0]*2)
		d = 0.0
		for x in X:
			q += (self.evalfuncphi(self.evalfuncH(x))*self.evalfuncI(self.evalfuncH(x), g)*self.evalfuncTheta(x))
			d += (self.evalfuncphi(self.evalfuncH(x))*self.evalfuncI(self.evalfuncH(x), g))

		print "qd: ",q,d, q/d
		return q/d
	

	def evalfunc2(self, X, k, z):
		N = len(X)
		r = np.array([0.0]*2)
		for x in X:
			r += self.evalfuncTheta(x)
		lk = float(k)**(-self.lbda)
		print ">>> lk: ", lk
		return (lk/N)*r+(1-lk)*z

	def evalDistParams1(self, X, k, g, zo):
		a = 0.5/float(k)
 		zn = a*self.evalfunc1(X, g)+(1-a)*self.evalfunc2(X, k, zo)
		print "zn: ", zn
		return zn
			
	def evalDistParams(self, X, k, g, zo):
		q = 0.0
		d = 0.0
		for x in X:
			q += (self.evalfuncphi(self.evalfuncH(x))*self.evalfuncI(self.evalfuncH(x), g)*x)
			d += (self.evalfuncphi(self.evalfuncH(x))*self.evalfuncI(self.evalfuncH(x), g))
		f = q/d
		#a = 0.5/float(k)
 		#zn = a*f+(1-a)*zo
 		zn = f
		return zn

	def findPbar(self, p, q, g, sl, N, c):
		e = self.eps
		if c == 0:
			return (-1, -1)
		np = p+((q-p)/2.0)
		sq = self.getQuantile(sl, 1.0-np)
		writelog(1,"In Pbar: " + str(sq) + "\n")
		if sq <= g + e/2.0:
			crp = (-1, -1)
			rp = self.findPbar(p, np, g, sl, N, c-1)
		else:
			crp = (0, np)
			rp = self.findPbar(np, q, g, sl, N, c-1)
		if rp[0] != -1:
			return rp
		return crp
		

			
	def mrasScheme(self, z0, g0, N0, iLmt=20000):
		(Nk, gk) = (N0, g0)
		self.theta0 = z0
		zk = z0
		pk = 0.1
		writelog(1, "In mrasScheme\n")
		for k in range(1, self.ITERCOUNT):
			if (Nk > iLmt):
				break
			writelog(1, "Iter: "+str(k) + ": " + str(Nk) + "\n")
#			clrscr(win, "["+str(k)+"]", 1, 6)
			Xk = self.getSamples(zk, Nk)
			writelog(1, "Xk:  " + str(Xk) + "\n")
			Vk = self.getFuncVals(Xk)
			writelog(1, "Vk:  " + str(np.sort(Vk)) + "\n")
			tgk = self.getQuantile(Vk, 1.0-pk)
			writelog(1, "tgk:  " + str(tgk) + "\n")
			#for t in range(len(Xk)):
			#	print Xk[t], ":", Vk[t]
			#print "==> ", tgk,"[",gk,"]"
			if tgk > gk + self.eps/2.0:
				gk = tgk
			else:
				rb = self.findPbar(0.0, pk, gk, Vk, Nk, 10)
				#print "Pbar: ", rb
				if rb[0] == 0:
					pk = rb[1]
				else:
					Nk = int(math.ceil(self.dlta*Nk))
					continue

			#zk1  = self.evalDistParams(Xk, k, gk, zk)
			zk1  = self.computeTheta(zk, Nk, Xk, Vk, gk, k)
			writelog(1, "New Theta: " + str(zk1[0]) + "\n")
			#tr = np.arange(-10,10,0.01)
			#plt.plot(tr, self.evalfuncF(tr, (zk[0],zk[1]-zk[0]*zk[0])))
			#plt.plot(tr, self.evalfuncF(tr, (zk[0],zk[1])))
			zk = zk1
			self.theta0_shr = 0.9

		return zk

	def mrasScheme1(self, n, p, g0, N0):
		(Nk, gk) = (N0, g0)
		zk = p
		pk = 0.01
		for k in range(1, self.ITERCOUNT):
			print "!!! ", k, n, zk, "Nk: ", Nk
			Xk = self.getSamples(np.array([n,zk]), Nk, 1.0/float(k))
			Vk = self.getFuncVals(Xk)
			#print Xk
			#print "Sorted Vk: ", np.sort(Vk)
			tgk = self.getQuantile(Vk, 1.0-pk)
			#for t in range(len(Xk)):
			#	print Xk[t], ":", Vk[t]
			#print "==> ", tgk,"[",gk,"]"
			if tgk > gk + self.eps/2.0:
				gk = tgk
			else:
				rb = self.findPbar(0.0, pk, gk, Vk, Nk, 10)
				if rb[0] == 0:
					pk = rb[1]
				else:
					Nk = int(math.ceil(self.dlta*Nk))
					continue

			zk1  = self.evalDistParams(Xk, k, gk, zk*float(n))
			#zk1  = self.computeTheta(zk, Nk, Xk, gk, k)
			zk = zk1/float(n)
		return (zk, gk)


class mdp(mrasd):
	def __init__(self, ssize, asize, kdim):
		mrasd.__init__(self)
		self.vi_thld = 0.01
		self.A = range(asize)
		self.S = range(ssize)
		self.asz = asize
		self.ssz = ssize
		sl = [0.0]*ssize
		asl = [sl[:]]*asize
		sasl = [asl[:]]*ssize
		self.P = scpy.array(sasl[:])
		self.g = scpy.array(sasl[:])
		self.gamma = 0.01
		self.lbda = 0.99
		self.kdim = kdim
		self.crtst = -1
		self.def_theta = None


	def norm(self, x, std):
		D = np.identity(self.ssz)
		for i in range(self.ssz):
			D[i][i] = std[i]
		return scpy.dot(x, scpy.dot(D,x))
		
	def projection(self, J, std):
		nJ = J[:]
		nJ.resize(len(nJ), 1)
		D = scpy.identity(self.ssz, dtype=np.longdouble)
		for i in range(self.ssz):
			D[i][i] = std[i]
		ph = scpy.array([[0.0]*self.kdim]*self.ssz, dtype=np.longdouble)
		for s in self.S:
			rv = self.evalftval(s)
			ph[s] += rv
		tph = scpy.transpose(ph)
		E = scpy.dot(scpy.dot(tph, D), ph)
		iE = linalg.pinv(E)
		pi = scpy.dot(scpy.dot(scpy.dot(ph, iE), tph), D)
		return scpy.dot(pi, nJ).flatten()


	def evalftval(self, x, theta=[]):
		if len(theta) == 0:
			theta = self.def_theta.flatten()

		fc = theta[0:len(theta):2]
		fz = theta[1:len(theta):2]
		rv = scpy.array([0.0]*self.kdim, dtype=np.longdouble)
		for i in range(self.kdim):
			try:
				pw = (-1.0)*np.power(fz[i],2.0)*np.power(float(x)-fc[i],2.0) 
				rv[i] = np.exp(pw, dtype=np.longdouble)
			except OverflowError:
				#writelog(1, "Exception1: >> " + str(x) + " " + str(fz[i]) + " " + str(fc[i]) + "\n")
				rv[i] = 0.0
			except FloatingPointError:
				rv[i] = 0.0
		
		return rv
			
	def evaldftval(self, theta, x, q):
		rv = scpy.array([0.0]*self.kdim, dtype=np.longdouble)
		if (q%2 == 0):
			rv[q/2] = self.evaldftval1(theta[q], theta[q+1], float(x))
		else:
			rv[q/2] = self.evaldftval2(theta[q-1], theta[q], float(x))
		return rv
		
	def evaldftval1(self, m, v, x):
		c = (2.0*(v**2.0)*(x-m))
		try:
			rv = c*scpy.exp(-(v**2.0)*((x-m)**2.0), dtype=np.longdouble)
		except OverflowError:
			writelog(1, "Exception2: >> " + str(x) + " " + str(v) + " " + str(m) + "\n")
			sys.exit(1)
		except FloatingPointError:
			writelog(1, " ####### zero val1 >> " + str(x) + " " + str(v) + " " + str(m) + "\n")
			rv = 0.0
		return rv

	def evaldftval2(self, m, v, x):
		c = (-(2.0*v)*((x-m)**2))
		try:
			rv = c*scpy.exp(-(v**2.0)*((x-m)**2.0), dtype=np.longdouble)
		except OverflowError:
			writelog(1, "Exception3: >> " + str(x) + " " + str(v) + " " + str(m) + "\n")
			sys.exit(1)
		except FloatingPointError:
			writelog(1, " ####### zero val2 >> " + str(x) + " " + str(v) + " " + str(m) + "\n")
			rv = 0.0
		return rv

	def genStochRow(self,n):
		sumv = Fraction(0.0)
		rv = [Fraction(0.0)]*n
		sv = np.random.permutation(n)
		for s in sv[0:n-1]:
			v = Fraction(scpy.random.uniform(0,1.0-sumv))
			sumv += v
			rv[s] = v
		rv[sv[n-1]] = 1.0-sumv
		return scpy.array(rv)

	def genP(self):
		for s in self.S:
			for a in self.A:
				rv = self.genStochRow(self.ssz)
				#rv = np.array([1.0/float(self.ssz)]*self.ssz)
				self.P[s][a] += rv


	def genG(self, bd):
		for s in self.S:
			for a in self.A:
				rv = bd*np.random.random_sample(self.ssz)
				self.g[s][a] += rv
		
			
	def bellman(self, J):
		av = [0.0]*len(self.A)
		tsa = [av[:]]*len(self.S)
		sa = np.array(tsa)
		u = np.array([0.0]*self.ssz)
		nJ = np.array([0.0]*self.ssz)
		for s in self.S:
			for a in self.A:
				g_u = scpy.dot(self.P[s][a],self.g[s][a])
				T_u_j = g_u +  self.gamma*(scpy.dot(self.P[s][a], J))
				sa[s][a] = T_u_j
			u[s] = np.argmin(sa[s])
			nJ[s] = np.min(sa[s])
		return (u, nJ)
	
	

	def bellman1(self, J):
		n = len(self.S)
		m = len(self.A)
		tJ = [0.0]*n
		for s in self.S:
			self.crtst = s
			#tJ[s] = self.mrasScheme(m, 0.5, 0, 40)
		
	def valueIter(self, J):
		err = 1.0
		nJ = oJ = J
		t = 0
		while (err > self.vi_thld):
			t += 1
			(u, nJ) = self.bellman(oJ)
			err = np.linalg.norm(nJ-oJ, ord=np.inf)
			#print "in loop: ", u, "err: ", err, nJ, oJ
			oJ = nJ
			if (t > 1000):
				break
	
		return (u,nJ)

	def policyEval(self, u):
		g_u = np.array([0.0]*self.ssz)
		P_u = np.array([[0.0]*self.ssz]*self.ssz)
		for s in self.S:
			g_u[s] = scpy.dot(self.P[s][u[s]],self.g[s][u[s]])
			P_u[s] += self.P[s][u[s]]
		I = np.identity(self.ssz)
		dP = (I-(self.gamma*P_u))
		J_u =  np.linalg.solve(dP, g_u)
		return J_u
		
	def policyIter(self, u):
		ou = nu = u
		err = 1.0
		while (err > self.vi_thld):
			ou = nu
			Ju = self.policyEval(ou)
			(nu, Jnu) = self.bellman(Ju)
			err = np.linalg.norm(Ju-Jnu, ord=np.inf)

		return (nu, Jnu)
	
	def conv_to_decimal(self, u, b):
		ul = len(u)
		mt = 1
		v = 0
		for i in range(ul):
			v += int(u[i])*mt
			mt *= b
		return v
		
	def conv_to_policy(self, d, b, ul):
		u = []
		t = d
		c = 0
		while ((t != 0) or (c < ul)):
			u.append(int(t%b))
			t /= b
			c += 1
		return u
	
	def evalfuncH(self, x):
		a = x
		s = self.crtst
		#u = self.conv_to_policy(x, len(self.A), len(self.S))
		Ju = self.policyEval(u)
		return Ju[s]


class sim:
	def __init__(self, mdpo, pcy, pf, s, lm):
		self.mdpo = mdpo
		self.P = mdpo.P
		self.apmf = pf
		self.current = s
		self.limit = lm
		self.time = 0
		self.randv = myrand()
		self.pcy = pcy

	def __iter__(self):
		return self

	def next(self):
		if self.time >= self.limit:
            		raise StopIteration
		else:
			s = self.current
			#a_pmf = self.apmf[s]
			#s_a = self.randv.getRandom(a_pmf, 1)[0] 
			s_a = self.pcy[s]
			s_pmf = self.P[s][s_a]
			ns = self.randv.getRandom(s_pmf, 1)[0]
			self.current = ns
			self.time += 1
			return (ns, self.mdpo.g[s][s_a][ns])
		return -1
	
class ValueEst:
	def __init__(self, mdpobj, s, u, std):
		self.mdp = mdpobj
		self.state = s
		self.policy = u
		self.stdst = std
		self.TDJv = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.MCmult = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.target = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.MCJ = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.MCstep = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.MCvisit = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.Jv = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		self.eJv = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)

	def MC(self, time, s, g):
		if (time < 100):
			bt = 0.1
		elif (time < 1000):
			bt = 0.01
		elif (time < 5000):
			bt = 0.01
		else:
			bt = 0.0001
		bt = 1.0/math.sqrt(time)
		self.MCstep[self.state] += 1.0 
		self.MCvisit[self.state] += 1.0
		if (self.MCvisit[self.state] == 2.0):
			#bt = 1.0/self.MCstep[self.state]
			self.MCJ[self.state] += bt*(self.target[self.state]-self.MCJ[self.state])
			self.MCvisit[self.state] = 1.0
			self.MCmult[self.state] = 1.0
			self.target[self.state] = 0
		try:
			self.target[self.state] += self.MCmult[self.state]*g
		except:
			self.target[self.state] += 0.0
		try:
			self.MCmult[self.state] *= self.mdp.gamma
		except:
			self.MCmult[self.state]  = 0
			

	def TD0(self, time, s, g):
		if (time < 100):
			bt = 0.1
		elif (time < 1000):
			bt = 0.01
		elif (time < 5000):
			bt = 0.01
		else:
			bt = 0.001
		dlta = g + self.mdp.gamma*self.TDJv[s] - self.TDJv[self.state]
		self.TDJv[self.state] += bt*dlta

	def TDlbda(self, time, s, g):
		inc = g+self.mdp.gamma*self.Jv[s]-self.Jv[self.state]
		#bt = 1.0/float(time)
		if (time < 100):
			bt = 0.1
		elif (time < 1000):
			bt = 0.01
		elif (time < 5000):
			bt = 0.01
		else:
			bt = 0.001
		
		for x in self.mdp.S:
			try:
				self.Jv[x] += bt*inc*self.eJv[x]
			except:
				writelog(1, "exception1!!!! \n\n")
				self.Jv[x] += 0.0
			try:
				self.eJv[x] *= self.mdp.gamma*self.mdp.lbda
			except:
				writelog(1, "Exception: " + str(self.eJv[x]) + " " + str(self.mdp.gamma*self.mdp.lbda) + "\n\n")
				self.eJv[x] = 0.0
		self.eJv[self.state] = 1.0


	def getnext(self, time, s, g):
		self.TD0(time, s, g)
		self.TDlbda(time, s, g)
		self.MC(time, s, g)
		self.state = s 

class Actor(mrasd):
	def __init__(self, mdpobj, stdobj):
		mrasd.__init__(self)
		self.role = "Actor"
		self.mdp = mdpobj
		self.std = stdobj

	def getSamples(self, z, sz):
		m = self.theta0_shr*z[0] + (1.0-self.theta0_shr)*self.theta0[0]
		cov = self.theta0_shr*z[1] + (1.0-self.theta0_shr)*self.theta0[1]
		return np.random.multivariate_normal(m, cov, sz)

	def getFuncVals3(self, X):
		writelog(1, "getFuncVals3.................\n")
		V = np.array([0.0]*len(X))
		for i in range(len(X)):
			if np.max(X[i]) >= 1.0 or np.min(X[i]) < 0:
				V[i] += 999999
				continue
				
			u = np.array(X[i]*self.mdp.asz, dtype=np.int)
			ist = scpy.random.random_integers(0, self.mdp.ssz-1)
			vst = ValueEstWithFuncAppr(self.mdp, ist, u, self.std, None)
			#ce_r = vst.CEMethod()
			vst.RLSTD_init()
			sm = sim(self.mdp, u, [], ist, 1000)
			for st in sm:
				vst.getnext(sm.time, st[0], st[1])
			J = []
			for k in range(self.mdp.ssz):
				#J.append(scpy.dot(self.mdp.evalftval(k), ce_r))
				J.append(scpy.dot(self.mdp.evalftval(k), vst.rlstd_rt))
			V[i] = scpy.linalg.norm(J)

		return (-1.0)*V

	def perform(self):
		m0 = scpy.array([0.5]*self.mdp.ssz, dtype=np.longdouble)
		cov0 = 0.05*scpy.identity(self.mdp.ssz, dtype=np.longdouble)
		writelog(1, "[Actor-CEMethod]\n")
		self.getFuncVals = self.getFuncVals3
		out = self.mrasScheme((m0, cov0), -500000.0, 50,400)
		theta = scpy.longdouble(scpy.random.multivariate_normal(out[0], out[1], 1))
		writelog(1, "Final params: " + str(theta[0]) + "\n")
		return theta[0]
		
	
		
class ValueEstWithFuncAppr(mrasd):
	def __init__(self, mdpobj, s, u, std, vJ):
		mrasd.__init__(self)
		self.rt = scpy.array([0.0]*mdpobj.kdim, dtype=np.longdouble)
		self.zt = scpy.array([0.0]*mdpobj.kdim, dtype=np.longdouble)
		self.mdp = mdpobj
		self.state = s
		self.policy = u
		self.stdst = std
		self.At = scpy.array([[0.0]*mdpobj.kdim]*mdpobj.kdim, dtype=np.longdouble)
		self.bt = scpy.array([0.0]*mdpobj.kdim, dtype=np.longdouble)
		self.vJ = vJ
		self.rlstd_rt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.rlstd_Ct = 900000.0*scpy.identity(self.mdp.kdim, dtype=np.longdouble)
		self.gtd2_wt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.gtd2_rt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.gtd2v2_wt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.gtd2v2_rt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.lstd_At = 0.0*scpy.identity(self.mdp.kdim, dtype=np.longdouble)
		self.lstd_bt = scpy.array([0.0]*mdpobj.kdim, dtype=np.longdouble)
		self.lspeN = 5000
		self.lspe_rt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.sL = [s]
		self.gL = []
		self.ppt = scpy.array([[0.0]*mdpobj.kdim]*mdpobj.kdim, dtype=np.longdouble)
		self.dpt1 = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.dpt2 = scpy.array([[0.0]*self.mdp.kdim]*self.mdp.kdim, dtype=np.longdouble)

	def getSamples(self, z, sz):
		m = self.theta0_shr*z[0] + (1.0-self.theta0_shr)*self.theta0[0]
		cov = self.theta0_shr*z[1] + (1.0-self.theta0_shr)*self.theta0[1]
		return np.random.multivariate_normal(m, cov, sz)


	def evalfuncH(self, theta, r):
		fval = 0.0
		u = self.policy
		for s in range(self.mdp.ssz):
			op1 = scpy.dot(self.mdp.evalftval(s, theta), r) 
			op2 = 0.0
			for t in range(self.mdp.ssz):
				op2 += (self.mdp.P[s][u[s]][t]*(self.mdp.g[s][u[s]][t] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(t, theta))))
			fval += (self.stdst[s]*((op1-op2)**2.0))
		return fval

	def evalfunc101_preprocess(self, theta):
		sN = 100000
		est = scpy.array([[0.0]*self.mdp.kdim]*self.mdp.kdim, dtype=np.longdouble)
		est3 = scpy.array([[0.0]*self.mdp.kdim]*self.mdp.kdim, dtype=np.longdouble)
		est2 = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		oldst = scpy.random.random_integers(0, self.mdp.ssz-1)
		sm = sim(self.mdp, self.policy, [], oldst, sN)
		for st in sm:
			phi_k = self.mdp.evalftval(oldst, theta)
			phi2_k = self.mdp.evalftval(st[0], theta)
			n_phi_k = np.reshape(phi_k, (self.mdp.kdim,1))
			n_phi2_k = np.reshape(phi2_k, (self.mdp.kdim,1))
			est += scpy.dot(n_phi_k, scpy.transpose(n_phi_k))
			est2 += (self.mdp.evalftval(oldst, theta)*st[1])
			est3 += scpy.dot(n_phi_k, scpy.transpose(n_phi2_k))
			#est3 += self.mdp.gamma*self.mdp.evalftval(st[0], theta)-self.mdp.evalftval(oldst, theta)
			oldst = st[0]
		self.ppt += (est/np.longdouble(sN))
		self.dpt1 += (est2/np.longdouble(sN))
		self.dpt2 += (est3/np.longdouble(sN))
		
	def evalfuncH101(self, theta, r):
		u = self.policy
		sN = 2000
		oldst = 0
		#sm = sim(self.mdp, self.policy, [], oldst, sN)
		#est = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		#for st in sm:
		#	dk = st[1] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(st[0], theta))-scpy.dot(r, self.mdp.evalftval(oldst, theta))
		#	est += (self.mdp.evalftval(oldst, theta)*dk)
		#	oldst = st[0]
		#fn_est = scpy.reshape(est,(self.mdp.kdim,1))/np.longdouble(sN)
		est = self.dpt1 + self.mdp.gamma*scpy.dot(self.dpt2, r)-scpy.dot(self.ppt,r)
		fn_est = scpy.reshape(est,(self.mdp.kdim,1))
		
		rv = scpy.dot(scpy.dot(scpy.transpose(fn_est), linalg.pinv(self.ppt)), fn_est)
		return rv
		
		
	def evalfuncH99test(self, theta, r):
		fval = 0.0
		u = self.policy
		J = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		TJ = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		#TpJ = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		for s in range(self.mdp.ssz):
			op1 = scpy.dot(self.mdp.evalftval(s, theta), r) 
		#pJ = self.mdp.projection(J, self.stdst)
			op2 = 0.0
			for t in range(self.mdp.ssz):
				op2 += (self.mdp.P[s][u[s]][t]*(self.mdp.g[s][u[s]][t] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(t, theta))))
				
			TJ[s] = op2
			J[s] = op1
		pTJ = self.mdp.projection(TJ, self.stdst)
		#fval = scpy.dot(self.stdst[0:np.size(self.stdst)-1], ((J-TpJ)**2.0))
		fval = scpy.dot(self.stdst[0:np.size(self.stdst)-1], ((J-pTJ)**2.0))
		return fval

	def evalfuncH99(self, theta, r):
		fval = 0.0
		u = self.policy
		J = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		TJ = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		#TpJ = scpy.array([0.0]*self.mdp.ssz, dtype=np.longdouble)
		for s in range(self.mdp.ssz):
			op1 = scpy.dot(self.mdp.evalftval(s, theta), r) 
		#pJ = self.mdp.projection(J, self.stdst)
			#op2 = 0.0
			#for t in range(self.mdp.ssz):
				#op2 += (self.mdp.P[s][u[s]][t]*(self.mdp.g[s][u[s]][t] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(t, theta))))
			for scnt in range(200):
				sm = sim(self.mdp, self.policy, [], s, 1)
				for st in sm:
					TJ[s] += (st[1] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(st[0], theta)))
				
			#	op2 += (self.mdp.P[s][u[s]][t]*(self.mdp.g[s][u[s]][t] + self.mdp.gamma*pJ[t]))
			TJ[s] = TJ[s]/float(200)
			J[s] = op1
		pTJ = self.mdp.projection(TJ, self.stdst)
		#fval = scpy.dot(self.stdst[0:np.size(self.stdst)-1], ((J-TpJ)**2.0))
		fval = scpy.dot(self.stdst[0:np.size(self.stdst)-1], ((J-pTJ)**2.0))
		return fval

	def evalfuncH2(self, theta, r):
		u = self.policy
		fval = 0.0
		for s in range(self.mdp.ssz):
			op1 = scpy.dot(self.mdp.evalftval(s, theta), r) 
			fval += (self.stdst[s]*((op1-self.vJ[s])**2.0))

		return fval

	def getFuncVals(self, X):
		V = scpy.array([0.0]*len(X), dtype=np.longdouble)
		oldst = scpy.random.random_integers(0, self.mdp.ssz-1)
		self.RLSTD2_init(len(X))
		sm = sim(self.mdp, self.policy, [], oldst, 500)
		for s in sm:
			#progressbar(win, "[ " + str(sm.time) + " ]", 10, 6)
			writelog(1, str(sm.time)+"\n")
			self.RLSTD2(time, oldst, s[0], s[1], X)
			oldst = s[0]

		for i in range(len(X)):
			V[i] = self.evalfuncH(X[i], self.rlstd2_rt[i])
		return (-1.0)*V


	def getFuncVals2(self, X):
		V = np.array([0.0]*len(X))
	#	tV = np.array([0.0]*len(X))
		for i in range(len(X)):
			V[i] = self.evalfuncH101(self.mdp.def_theta.flatten(), X[i])
		#	tV[i] = self.evalfuncH99test(self.mdp.def_theta.flatten(), X[i])
		
		#writelog(1, "V"+"\n")
		#writelog(1, str(V)+"\n")
		#writelog(1, "tV"+"\n")
		#writelog(1, str(tV)+"\n")
		return (-1.0)*V

	def computeTheta1(self, T, N, S, g, k):
		thsz = 2*self.mdp.kdim
		mean = np.array([0.0]*thsz)
		var = [0.0]*thsz
		cov = np.array([[0.0]*thsz]*thsz)
		for q in range(thsz):
			op1 = 0.0
			cN = 0.0
			for i in range(N):
				if (S[i] >= g):
					op1 += T[i][q]
					cN += 1.0
			mean[q] = op1/cN
		for q in range(thsz):
			op2 = 0.0
			cN = 0.0
			for i in range(N):
				if (S[i] >= g):
					op2 += ((T[i][q]-mean[q])**2.0)
					cN += 1.0
			var[q] = op1/cN
			cov[q][q] = var[q]
		
		return (mean, cov)
		

	
	def LSTD(self, time, s, g):
		f1 = self.mdp.evalftval(self.state)
		f2 = self.mdp.evalftval(s)
		f3 = f1-self.mdp.gamma*f2
		f1.resize(len(f1),1)
		f3.resize(1, len(f2))
		self.lstd_At += scpy.dot(f1,f3)
		self.lstd_bt += g*self.mdp.evalftval(self.state)


	def GTD2(self, time, s, g):
		#bt = 1.0/(float(time)**0.52)
		#ct = 1.0/(float(time)**0.7)
		if (time < 100):
			bt = 0.1
		elif (time < 1000):
			bt = 0.01
		elif (time < 5000):
			bt = 0.01
		else:
			bt = 0.001
		ct = bt
		f1 = self.mdp.evalftval(self.state)
		f2 = self.mdp.evalftval(s)
		td = g + scpy.dot((self.mdp.gamma*f2-f1), self.gtd2_rt)
		a = scpy.dot(f1, self.gtd2_wt)
		self.gtd2_rt += ct*a*(f1-self.mdp.gamma*f2)
		self.gtd2_wt += bt*(td-a)*f1

	def GTD2v2(self, time, s, g):
		mlt = 1.0
		if (time < 100):
			ct = 0.01
			bt = 0.1
		elif (time < 1000):
			ct = 0.0009
			bt = 0.008
		elif (time < 5000):
			bt = 0.001
		else:
			mlt = 5000.0
			ct = 0.0001
			bt = 0.001
		bt = 1.0/(float(time)**0.6)
		ct = 1.0/float(time)
		f1 = self.mdp.evalftval(self.state)
		f2 = self.mdp.evalftval(s)
		td = g + scpy.dot((self.mdp.gamma*f2-f1), self.gtd2v2_rt)
		a = scpy.dot(f1, self.gtd2v2_wt)
		try:
			self.gtd2v2_rt += ct*(td*f1-a*self.mdp.gamma*f2)
			self.gtd2v2_wt += bt*(td-a)*f1
		except FloatingPointError:
			writelog(1,"Errorr: !!! : " + str(self.gtd2v2_wt) + "\n\n")
			writelog(1,"Errorr: !!! : " + str(self.gtd2v2_rt) + "\n\n")
			sys.exit(0)
		
		
	def LSPE(self, time):
		d = 0.0
		b = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		A = scpy.array([[0.0]*self.mdp.kdim]*self.mdp.kdim, dtype=np.longdouble)
		itr = range(self.lspeN)
		itr.reverse()
		for i in itr:
			f1 = self.mdp.evalftval(self.sL[i])
			f2 = self.mdp.evalftval(self.sL[i+1])
			v = scpy.dot(self.lspe_rt, f1)
			d = self.mdp.gamma*self.mdp.lbda*d + self.gL[i] + self.mdp.gamma*scpy.dot(self.lspe_rt, f2)-v
			f1c = f1[:]
		 	b = b+((v+d)*f1)
			f1.resize(len(f1),1)
			f1c.resize(1, len(f1c))
			A = A+scpy.dot(f1,f1c)
		b.resize(len(b), 1)
		nrt = scpy.dot(linalg.pinv(A), b).flatten()
		ct = float(self.lspeN)/float(time)
		self.lspe_rt = self.lspe_rt + ct*(nrt-self.lspe_rt)
			
	def RLSTD_init(self):
		self.rlstd_rt = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		self.rlstd_Ct = 900000.0*scpy.identity(self.mdp.kdim, dtype=np.longdouble)


	def RLSTD2_init(self, N):
		self.rlstd2_rt = scpy.array([[0.0]*self.mdp.kdim]*N, dtype=np.longdouble)
		self.rlstd2_Ct = scpy.array([900000.0*np.identity(self.mdp.kdim, dtype=np.longdouble)]*N)
		
	def RLSTD2(self, time, s1, s2, g, Tlst):
		for i in range(len(Tlst)):
			f1 = self.mdp.evalftval(s1, Tlst[i])
			f2 = self.mdp.evalftval(s2, Tlst[i])
			b = scpy.dot((f1-self.mdp.gamma*f2), self.rlstd2_Ct[i])
			a = 1.0 + scpy.dot(b,f1)
			v = scpy.dot(self.rlstd2_Ct[i], f1)
			td = g + self.mdp.gamma*scpy.dot(f2, self.rlstd2_rt[i])-scpy.dot(f1,self.rlstd2_rt[i])
			try:
				self.rlstd2_rt[i] += (td/a)*v
			except FloatingPointError:
				writelog(1,"Errorr RLSTD2: !!! : " + str(i) + "\n\n")
				sys.exit(0)
			
			v.resize(len(v), 1)
			b.resize(1, len(b))
			self.rlstd2_Ct[i] -= ((scpy.dot(v,b))/a)


	def RLSTD(self, time, s1, s2, g):
		f1 = self.mdp.evalftval(s1)
		f2 = self.mdp.evalftval(s2)
		b = scpy.dot((f1-self.mdp.gamma*f2), self.rlstd_Ct)
		a = 1.0 + scpy.dot(b,f1)
		v = scpy.dot(self.rlstd_Ct, f1)
		td = g + self.mdp.gamma*scpy.dot(f2, self.rlstd_rt)-scpy.dot(f1,self.rlstd_rt)
		try:
			self.rlstd_rt += (td/a)*v
		except FloatingPointError:
			writelog(1,"Errorr: !!! : " + str(td) + " " + str(a) + " " + str(v) + "\n\n")
			writelog(1,"Errorr: !!! : " + str(b) + " " + str(f1) + " " + str(f2) + "\n\n")
			writelog(1,"Errorr: !!! : " + str(self.rlstd_Ct) + "\n\n")
			sys.exit(0)
			
		v.resize(len(v), 1)
		b.resize(1, len(b))
		self.rlstd_Ct -= ((scpy.dot(v,b))/a)
				

	
	def CEMethod(self):
		fval = 0.0
		u = self.policy
		m0 = scpy.array([0.0]*self.mdp.kdim, dtype=np.longdouble)
		cov0 = 500.0*scpy.identity(self.mdp.kdim, dtype=np.longdouble)
		self.evalfunc101_preprocess(self.mdp.def_theta.flatten())
		writelog(1, "[CEMethod]\n")
		self.getFuncVals = self.getFuncVals2
		out = self.mrasScheme((m0, cov0), -500000.0, 4000)
		theta = scpy.longdouble(scpy.random.multivariate_normal(out[0], out[1], 1))
		writelog(1, "Final params: " + str(theta[0]) + "\n")
		return theta[0]
		
		
 
	def getnext(self, time, s, g):
		try:
			dt = g+self.mdp.gamma*scpy.dot(self.mdp.evalftval(s),self.rt)-scpy.dot(self.mdp.evalftval(self.state),self.rt)
		except FloatingPointError:
			writelog(1, "rt: "+ str(self.rt) + "\n\n")
			sys.exit(0)

			
		bt = 1.0/(float(time)**0.6)
		self.zt = self.mdp.gamma*self.mdp.lbda*self.zt+self.mdp.evalftval(self.state)
		self.rt = self.rt+bt*dt*self.zt

		dphi = self.mdp.evalftval(self.state) - self.mdp.gamma*self.mdp.evalftval(s)
		dphi.resize(1,self.mdp.kdim)
		czt = scpy.array(self.zt[:], dtype=np.longdouble)
		czt.resize(self.mdp.kdim,1)
		self.At = self.At + scpy.dot(czt,dphi)
		self.bt = self.bt + g*self.zt

		self.RLSTD(time, self.state, s, g)
		self.GTD2(time, s, g)
		self.GTD2v2(time, s, g)
		self.LSTD(time, s, g)
		self.sL.append(s)
		self.gL.append(g)
		if (time > 0 and time % self.lspeN == 0):
			self.LSPE(time)
			self.sL = [s]
			self.gL = []
		self.state = s 

		
	def getTDr(self, s):
		return self.stInfo[s][2]

	def gettTDr(self):
		return self.rt

	def getLSTDr(self,s):
		cbt = scpy.array(self.sbt[s][:], dtype=np.longdouble)
		cbt.resize(self.mdp.kdim,1)
		rv = scpy.dot(linalg.pinv(self.sAt[s]), cbt)
		return rv.flatten()

	def gettLSTDr(self):
		cbt = scpy.array(self.bt[:], dtype=np.longdouble)
		cbt.resize(self.mdp.kdim,1)
		rv = scpy.dot(linalg.pinv(self.At), cbt)
		return rv.flatten()

	def getLSTD0r(self):
		cbt = scpy.array(self.lstd_bt[:], dtype=np.longdouble)
		cbt.resize(self.mdp.kdim,1)
		rv = scpy.dot(linalg.pinv(self.lstd_At), cbt)
		return rv.flatten()

	def evalfn1(self, s, r, dr, theta, q):
		return (scpy.dot(dr, self.mdp.evalftval(s, theta))+scpy.dot(r, self.mdp.evaldftval(theta, s, q)))		

	def estimateFn1(self, s, r, dr, theta, q):
		val = 0.0
		N = 3000
		u = self.policy
		writelog(3, "State: " + str(s) + "\n")
		#pm = np.array([0.0]*self.mdp.ssz)
		#for i in range(N):
		#	sm = sim(self.mdp, u, [], s, 1)
		#	for sp in sm:
		#		val += self.evalfn1(sp[0], r, dr, theta, q)
		#		pm[sp[0]] += 1.0
		
		val = 0.0
		for t in self.mdp.S:
			val += (self.mdp.P[s][u[s]][t]*self.evalfn1(t, r, dr, theta, q))
				
		#writelog(3, "Est pmf: " + str(pm/float(N)) + "\n")
		#try:
		#	rv = val/float(N)
		return val
		#except FloatingPointError:
		#	return 0.0

	def estimateFn2(self, s, r, theta, q):
		val = 0.0
		N = 3000
		u = self.policy
		#for i in range(N):
		#	sm = sim(self.mdp, u, [], s, 1)
		#	for sp in sm:
		#		val += (sp[1] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(sp[0], theta)))
		#return val/float(N)
		val = 0.0
		for t in self.mdp.S:
			val += (self.mdp.P[s][u[s]][t]*(self.mdp.g[s][u[s]][t] + self.mdp.gamma*scpy.dot(r, self.mdp.evalftval(t, theta))))
		return val

	def getdAdb(self, s1, s2, g, theta, q):
		diff_phit = self.mdp.evalftval(s1, theta) - self.mdp.gamma*self.mdp.evalftval(s2, theta)
		diff_phit.resize(1, self.mdp.kdim)
		diff_dphit = self.mdp.evaldftval(theta, s1, q) - self.mdp.gamma*self.mdp.evaldftval(theta, s2, q)
		diff_dphit.resize(1, self.mdp.kdim)
		phit = self.mdp.evalftval(s1, theta)
		phit.resize(len(phit), 1)
		dphit = self.mdp.evaldftval(theta, s2, q)
		dphit.resize(len(dphit), 1)
			
		dAtinc = scpy.dot(dphit, diff_phit) + scpy.dot(phit, diff_dphit)
		try:
			dbtinc = g*self.mdp.evaldftval(theta, s1, q)
		except FloatingPointError:
			dbtinc = 0.0
		return (dAtinc, dbtinc)

	def getdAdb1(self, s, g, oldst, theta, q, dzt):
		phit = self.mdp.evalftval(oldst, theta) - self.mdp.evalftval(s, theta)
		phit.resize(1, self.mdp.kdim)
		dphit = np.array([0.0]*self.mdp.kdim)	
		dphit = self.mdp.evaldftval(theta, oldst, q) - self.mdp.evaldftval(theta, s, q)
		dphit.resize(1, self.mdp.kdim)
		cdzt = dzt[:]
		cdzt.resize(self.mdp.kdim, 1)
		czt = self.zt[:]
		czt.resize(self.mdp.kdim, 1)
			
		dAtinc = scpy.dot(cdzt, phit) + scpy.dot(czt, dphit)
		try:
			dbtinc = dzt*g
		except FloatingPointError:
			dbtinc = 0.0
		dztinc = self.mdp.evaldftval(theta, oldst, q)
		return (dAtinc, dbtinc, dztinc)
	

	def getdr(self, dAt, dbt):
		#invAt = np.linalg.inv(self.At)
		invAt = self.rlstd_Ct
		dr12 = scpy.dot(invAt, self.bt)
		dr13 = scpy.dot(dAt, dr12) 
		dr1 = scpy.dot(invAt, dr13)
		writelog(3, "dr1: " + str(dr1) + "\n")
		dr2 = scpy.dot(invAt, dbt)
		writelog(3, "dr2: " + str(dr2) + "\n")
		return dr2-dr1
		

	def getdJ(self, s, theta, r, dr, q):
		u = self.policy
		#dphit = np.array([0.0]*self.mdp.kdim)	
		dphit = self.mdp.evaldftval(theta, s, q)
		writelog(3, "s: "+str(s)+"\n")
		writelog(3, "dr: "+str(dr)+"\n")
		writelog(3, "r: "+str(r)+"\n")
		writelog(3, "q: " + str(q) + " dphit: "+str(dphit)+"\n")
	 	dJ1 = scpy.dot(dr, self.mdp.evalftval(s, theta)) + scpy.dot(r, dphit)
		writelog(3, "dJ1: "+str(dJ1)+"\n")
		dJ2 = self.estimateFn1(s, r, dr, theta, q)
		writelog(3, "dJ2: "+str(dJ2)+"\n")
		dJ = dJ1 - self.mdp.gamma*dJ2
		return dJ
		
	def getdS(self, theta, r, dr, q):
		std = self.stdst
		dS = 0.0
		for s in self.mdp.S:
			dJ = self.getdJ(s, theta, r, dr, q)
			writelog(3,"dJ: "+str(dJ)+"\n")
			writelog(3,"r: "+str(r)+"\n")
			writelog(3,"estFn2: "+str(self.estimateFn2(s, r, theta, q))+"\n")
			J = scpy.dot(r, self.mdp.evalftval(s, theta))-self.estimateFn2(s, r, theta, q)
			writelog(3,"J: "+str(J)+"\n")
			writelog(3,"std[s]: "+str(std[s])+"\n")
			try:
				dS += std[s]*J*dJ
			except FloatingPointError:
				dS += 0.0
		writelog(3,"dS: "+str(dS)+"\n")
		return 2.0*dS

	def getGradS(self, theta, rt):
		thsz = 2*self.mdp.kdim
		std = self.stdst
		u = self.policy
		dAt = np.array([[[0.0]*self.mdp.kdim]*self.mdp.kdim]*thsz)
		dbt = np.array([[0.0]*self.mdp.kdim]*thsz)
		dzt = np.array([[0.0]*self.mdp.kdim]*thsz)
		
		ist = np.random.randint(0,self.mdp.ssz)
		self.zt = np.array([0.0]*self.mdp.kdim)

		ist = 1
		#np.random.seed(10)
		sm = sim(self.mdp, u, [], ist, 100000)
		
		self.At = np.array([[0.0]*self.mdp.kdim]*mdpobj.kdim)
		self.bt = np.array([0.0]*self.mdp.kdim)
		oldst = ist
		self.RLSTD_init()
		pi = np.array([0.0]*self.mdp.ssz)
		writelog(3, "dAt: ---> " + str(dAt)+"\n\n")
		writelog(3, "dbt: ---> " + str(dbt)+"\n\n")
		writelog(3, "dbt: ---> " + str(self.bt)+"\n\n")
		writelog(3, "rt: ---> " + str(self.rlstd_rt)+"\n\n")
		writelog(3, "Ct: ---> " + str(self.rlstd_Ct)+"\n\n")
		for s in sm:
			#clrscr(win, " ", 40, 6)
			#progressbar(win, "[ " + str(sm.time) + " ]", 40, 6)
			writelog(1, "[" + str(sm.time) + "] Stte: " + str(s[0]) + "\n\n")
			for k in range(thsz):
				inc = self.getdAdb(oldst, s[0], s[1], theta, k)
				dAt[k] += inc[0]
				dbt[k] += inc[1]
				#try:
				#	dzt[k] = self.mdp.gamma*self.mdp.lbda*dzt[k] + inc[2]
				#except FloatingPointError:
				#	dzt[k] = inc[2]
			self.RLSTD(time, oldst, s[0], s[1])
			pi[s[0]] += 1.0
			#self.zt = self.mdp.gamma*self.mdp.lbda*self.zt+self.mdp.evalftval(oldst, theta)
			#dphi = self.mdp.evalftval(oldst, theta) - self.mdp.gamma*self.mdp.evalftval(s[0], theta)
			#dphi.resize(1,self.mdp.kdim)
			#czt = np.array(self.zt[:])
			#czt.resize(self.mdp.kdim,1)
			#self.At = self.At + scpy.dot(czt,dphi)
			self.bt += s[1]*self.mdp.evalftval(oldst, theta)
			oldst = s[0]
		
		writelog(3, "stdPI: ---> " + str(pi/5000.0)+"\n\n")
		writelog(3, "dAt: ---> " + str(dAt)+"\n\n")
		writelog(3, "dbt: ---> " + str(dbt)+"\n\n")
		#writelog(3, "dzt: ---> " + str(dzt)+"\n\n")
		dr = np.array([[0.0]*self.mdp.kdim]*thsz)
		
		r = self.rlstd_rt

		writelog(1,"new r: " + str(r)+"\n")
	#	return (1, np.array([0.0]*thsz))
		dS = []
		for q in range(thsz):
			dr[q] += self.getdr(dAt[q], dbt[q])
			writelog(3, str(q) + " > " + str(dr[q]) + "\n\n")
			dSq = self.getdS(theta, r, dr[q], q)
			writelog(1, "dS[" + str(q) + "] : " + str(dSq)+"\n\n")
			dS.append(dSq)
		writelog(1,"Gradient: " + str(dS)+"\n")

		#return (1, np.array([0.0]*thsz))
		return (0, np.array(dS))
		#return []
	
	def getOptBasis(self, r, vJ):
		writelog(1,"r: " + str(r)+"\n")
		thsz = 2*self.mdp.kdim
		th0 = np.array([0.0]*thsz)
		th0 = dth.flatten()[:]
		err = 1.0
		k = 1
		while ( k < 30):
			#clrscr(win, "[ " + str(k) + " ]", 30, 6)
			writelog(1,"k : " + str(k)+"\n")
			writelog(1,"theta: " + str(th0)+"\n")
			gt = 1.0/float(k**0.55)
			(status, gradS) = self.getGradS(th0, r)
			
			if (status == 0):
				th1 = th0 - gt*gradS
				rlstds = []
				for ms in range(self.mdp.ssz):
					rlstds.append(scpy.dot(self.mdp.evalftval(ms, th0), self.rlstd_rt))
				err4 = math.sqrt(self.mdp.norm(vJ-rlstds, self.stdst))
				display(win, str(err4), 2, 7+k);
			else:
				continue
			
			err = np.linalg.norm(th1-th0)
			writelog(1,"k: " + str(k) + "err val: " + str(err)+"\n")
			#if (err <= 0.01):
			#	break
			th0 = th1
			k += 1
		return (k, th1)


	def getOptBasis2(self):
		thsz = 2*self.mdp.kdim
		m0 = scpy.array([0.0]*thsz, dtype=np.longdouble)
		cov0 = 10000.0*scpy.identity(thsz, dtype=np.longdouble)
		writelog(1, "In getOptBasis2\n")
		out = self.mrasScheme((m0, cov0), -100000.0, 300, 700)
		theta = scpy.longdouble(scpy.random.multivariate_normal(out[0], out[1], 1))
		writelog(1, "Final params: " + str(theta[0]) + "\n")
		return theta[0]

def test1():
	plt.xlim(0,100)
	plt.ylim(0,4)

	rv = np.array([0.0]*dp1.kdim)
	t4 = np.arange(0,100,1.0)
	for i in range(dp1.kdim):
		plt.plot(t4, np.exp(-dp1.fzi[i]*((t4-dp1.fci[i])**2.0)), 'r')
	plt.axhline(y=0.0, color='k')
	plt.axvline(x=0.0, color='k')
	plt.show()

def progressbar(win, i, x, y):
	scrLock.acquire()
	win.addch(y,x,'-')
	win.addstr(y, x+5, str(i))
	win.refresh()
	win.addch(y,x,'\\')
	win.refresh()
	win.addch(y,x,'|')
	win.refresh()
	win.addch(y,x,'/')
	win.refresh()
	scrLock.release()

def clrscr(win, msg, x, y):
	scrLock.acquire()
	win.addstr(y, x, " "*20)
	win.addstr(y, x, msg)
	win.refresh()
	scrLock.release()

def display(win, msg, x, y):
	scrLock.acquire()
	win.addstr(y, x, " "*20)
	win.addstr(y, x, msg)
	win.refresh()
	scrLock.release()


def getStationaryDist(dp1, vu):
	lA = scpy.array([[0.0]*(dp1.ssz+1)]*(dp1.ssz+2), dtype=np.longdouble)
	cA = scpy.array([[0.0]*(dp1.ssz)]*(dp1.ssz), dtype=np.longdouble)

	for s in dp1.S:
		for nsi in dp1.S:
			cA[s][nsi] = dp1.P[s][vu[s]][nsi]
	cAt = scpy.transpose(cA)
	for s in dp1.S:
		for nsi in dp1.S:
			lA[s][nsi] = cAt[s][nsi]
	for nsi in dp1.S:
		lA[s+1][nsi] = 1.0
	for i in range(dp1.ssz+1):
		lA[i][i] = lA[i][i]-1.0
	lA[s+2][dp1.ssz] = 1.0
	lb = np.array([0.0]*(dp1.ssz+2))
	lb[dp1.ssz+1] = 1.0
	stn = linalg.lstsq(lA, lb)
	return stn[0]


def prodlist(ls):
	prod = 1.0
	for s in ls:
		prod *= s
	return prod

def printc(V1, V2):
	print "[ ",
	for x in range(len(V1)):
		if (math.fabs(V1[x] - V2[x]) < 0.05):
			print BGGREEN, V1[x], ENDC, 
		else:
			print BGRED, V1[x], ENDC,
	print ENDC, "]", ENDC

#stdscr = curses.initscr()
#curses.start_color()
#curses.noecho()
#curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
np.seterr(all='raise')
scpy.seterr(all='raise')
fptr = open('log', 'w')
outp = output()
mdpobj = mdp(50, 20, 20)
mdpobj.genP()
mdpobj.genG(10.0)
J0 = np.array([0]*mdpobj.ssz)
(vu, vJ) = mdpobj.valueIter(J0)
(pu, pJ) = mdpobj.policyIter(J0)
std = getStationaryDist(mdpobj, vu)
begin_x = 10; begin_y = 5
height = 45; width = 75
#win = curses.newwin(height, width, begin_y, begin_x)


class RL1(threading.Thread):
	def __init__(self, threadID, name, vu, vJ, pu, pJ, std):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.fptr = open('log2', 'w')
		self.dp1 = mdpobj
		self.apmf = np.array([[1.0/4.0]*4]*self.dp1.ssz)
		self.vu = vu
		self.vJ = vJ
		self.pu = pu
		self.pJ = pJ
		self.std = std
		self.win = None


	def run(self):
		threadLock1.acquire()
		self.win = win
		itrcnt = 300000
		for i in range(1):
			clrscr(self.win, "["+str(i)+"]", 1, 1)
			sm = sim(self.dp1, self.vu, self.apmf, i, itrcnt)
			vst = ValueEst(self.dp1, i, self.vu, self.std)
			pi = np.array([0.0]*self.dp1.ssz)
			for s in sm:
				progressbar(self.win, "[ " + str(sm.time) + " ]", 10, 1)
				pi[s[0]] += 1.0
				vst.getnext(sm.time, s[0], s[1])
				err1 = math.sqrt(self.dp1.norm(vst.Jv-self.vJ, self.std))
				err2 = math.sqrt(self.dp1.norm(vst.TDJv-self.vJ, self.std))
				err3 = math.sqrt(self.dp1.norm(vst.MCJ-self.vJ, self.std))
				display(self.win, str(err1), 2, 2);display(self.win, str(err2), 2, 3);display(self.win, str(err3), 2, 4)
			self.estsd = pi/float(itrcnt)
			outp.store1(self.dp1, self.estsd, self.std, self.vu, self.vJ, self.pu, self.pJ, vst.Jv, vst.TDJv, vst.MCJ)

		self.fptr.close()
		threadLock1.release()
		


class RL2(threading.Thread):
	def __init__(self, threadID, name, vu, vJ, pu, pJ, std):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.fptr = open('log3', 'w')
		self.mdp = mdpobj
		self.apmf = np.array([[1.0/4.0]*4]*self.mdp.ssz)
		self.vu = vu
		self.vJ = vJ
		self.pu = pu
		self.pJ = pJ
		self.std = std
		self.win = None
		self.vst = None

	def evalPolicy(self):
		tds = []
		lstds = []
		rlstds = []
		gtd2s = []
		gtd2v2s = []
		lstd0 = []
		lspes = []
		#lstdsop = []
		pvJ = self.mdp.projection(self.vJ, self.std)
		for i in range(self.mdp.ssz):
			tds.append(scpy.dot(self.mdp.evalftval(i), self.vst.gettTDr()))
			lstds.append(scpy.dot(self.mdp.evalftval(i), self.vst.gettLSTDr()))
			rlstds.append(scpy.dot(self.mdp.evalftval(i), self.vst.rlstd_rt))
			gtd2s.append(scpy.dot(self.mdp.evalftval(i), self.vst.gtd2_rt))
			gtd2v2s.append(scpy.dot(self.mdp.evalftval(i), self.vst.gtd2v2_rt))
			lstd0.append(scpy.dot(self.mdp.evalftval(i), self.vst.getLSTD0r()))
			lspes.append(scpy.dot(self.mdp.evalftval(i), self.vst.lspe_rt))
			#lstdsop.append(scpy.dot(dp1.evalftval(i), vst.gettLSTDr()))
		outp.store2(self.mdp, self.std, self.vu, self.vJ, pvJ, tds, lstds, rlstds, gtd2s, gtd2v2s, lstd0, lspes)
		err1 = math.sqrt(self.mdp.norm(self.vJ-pvJ, self.std))
		err2 = math.sqrt(self.mdp.norm(self.vJ-tds, self.std))
		err3 = math.sqrt(self.mdp.norm(self.vJ-lstds, self.std))
		err4 = math.sqrt(self.mdp.norm(self.vJ-rlstds, self.std))
		err5 = math.sqrt(self.mdp.norm(self.vJ-lspes, self.std))
		#display(self.win, str(err1), 2, 7);display(self.win, str(err2), 2, 8);display(self.win, str(err3), 2, 9)
		#display(self.win, str(err4), 2, 10);display(self.win, str(err5), 2, 11)

	def run(self):
		threadLock2.acquire()
		self.win = win
		itrcnt = 120000
		
		self.vu = [ 3,  0,  2,  0,  2,  0,  1,  2,  1,  2,  3,  1,  0,  3,  2,  1,  3,  3, 2,  1]
		pmu = [1.0/float(self.mdp.ssz)]*self.mdp.ssz
		for s in range(self.mdp.ssz):
			#self.mdp.P[s][self.vu[s]] = myP[s]
			self.mdp.P[s][self.vu[s]] = pmu
			writelog(1, str(s) + ": " + str(self.mdp.P[s][self.vu[s]]) + "\n")
			writelog(1, str(s) + ": " + str(sum(self.mdp.P[s][self.vu[s]])) + "\n")
			writelog(1, str(self.mdp.g[s][self.vu[s]]) + "\n")

		writelog(1, str(self.vu) + "\n")
		self.std = getStationaryDist(self.mdp, vu)
			
		for oi in range(1):
			clrscr(self.win, "["+str(oi)+"]", 1, 6)
			ist = 1
			self.vst = ValueEstWithFuncAppr(self.mdp, ist, self.vu, self.std)
			(k, thf) = self.vst.getOptBasis(self.vst.rlstd_rt, self.vJ)
			thf.resize(len(thf)/2, 2)
			#ist = np.random.randint(0,self.mdp.ssz)
			self.vst.RLSTD_init()
			sm = sim(self.mdp, self.vu, self.apmf, ist, itrcnt)
			for s in sm:
				progressbar(self.win, "[ " + str(sm.time) + " ]", 10, 6)
				self.vst.getnext(sm.time, s[0], s[1])
			rlstds = []
			for ms in range(self.mdp.ssz):
				rlstds.append(scpy.dot(self.mdp.evalftval(ms), self.vst.rlstd_rt))
			err4 = math.sqrt(self.mdp.norm(self.vJ-rlstds, self.std))
			display(win, str(err4), 2, 7+k+2);
			self.evalPolicy()
			curses.beep()
			win.getkey()

			#writelog(1, "start state: " + str(ist)+"\n")
			#writelog(1, str(self.vst.gettLSTDr())+"\n")
			#writelog(1, str(self.vst.rlstd_rt) + "\n")
			#writelog(1, str(self.vst.getLSTD0r()) + "\n") 
			#writelog(1, str(self.vst.rlstd_Ct) + "\n") 
			#writelog(1, str(np.linalg.inv(self.vst.lstd_At)) + "\n")
			#tds = []
			#for i in range(self.mdp.ssz):
			#	tds.append(scpy.dot(self.mdp.evalftval(i), self.vst.rlstd_rt))
			#writelog(1, str(tds) + "\n")

		self.fptr.close()
		threadLock2.release()



class RL3(threading.Thread):
	def __init__(self, threadID, name, vu, vJ, pu, pJ, std):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.fptr = open('log3', 'w')
		self.mdp = mdpobj
		self.apmf = np.array([[1.0/4.0]*4]*self.mdp.ssz)
		self.vu = vu
		self.vJ = vJ
		self.pu = pu
		self.pJ = pJ
		self.std = std
		self.win = None
		self.vst = None
		self.rlstd_err = None
		self.lstd0_err = None
		self.lstdlbda_err = None
		self.gtd2_err = None
		self.gtd2v2_err = None
		self.tdlbda_err = None
		self.actor = None

	def evalPolicy(self, ce_r):
		tds = []
		lstds = []
		rlstds = []
		gtd2s = []
		gtd2v2s = []
		lstd0 = []
		lspes = []
		lsce = []

		pvJ = self.mdp.projection(self.vJ, self.std)
		for i in range(self.mdp.ssz):
			tds.append(scpy.dot(self.mdp.evalftval(i), self.vst.gettTDr()))
			lstds.append(scpy.dot(self.mdp.evalftval(i), self.vst.gettLSTDr()))
			rlstds.append(scpy.dot(self.mdp.evalftval(i), self.vst.rlstd_rt))
			gtd2s.append(scpy.dot(self.mdp.evalftval(i), self.vst.gtd2_rt))
			gtd2v2s.append(scpy.dot(self.mdp.evalftval(i), self.vst.gtd2v2_rt))
			lstd0.append(scpy.dot(self.mdp.evalftval(i), self.vst.getLSTD0r()))
			lspes.append(scpy.dot(self.mdp.evalftval(i), self.vst.lspe_rt))
			lsce.append(scpy.dot(self.mdp.evalftval(i), ce_r))
		outp.store2(self.mdp, self.std, self.vu, self.vJ, pvJ, tds, lstds, rlstds, gtd2s, gtd2v2s, lstd0, lspes, ce_r, self.vst.rlstd_rt, self.vst.getLSTD0r(), self.mdp.def_theta, lsce)

		err1 = math.sqrt(self.mdp.norm(self.vJ-pvJ, self.std))
		err2 = math.sqrt(self.mdp.norm(self.vJ-tds, self.std))
		err3 = math.sqrt(self.mdp.norm(self.vJ-lstds, self.std))
		err4 = math.sqrt(self.mdp.norm(self.vJ-rlstds, self.std))
		err5 = math.sqrt(self.mdp.norm(self.vJ-lspes, self.std))
		#display(self.win, str(err1), 2, 7);display(self.win, str(err2), 2, 8);display(self.win, str(err3), 2, 9)
		#display(self.win, str(err4), 2, 10);display(self.win, str(err5), 2, 11)

	def evalError(self):
		tds = []
		lstds = []
		rlstds = []
		gtd2s = []
		gtd2v2s = []
		lstd0 = []
		lspes = []
		pvJ = self.mdp.projection(self.vJ, self.std)
		
		for i in range(self.mdp.ssz):
			tds.append(scpy.dot(self.mdp.evalftval(i), self.vst.gettTDr()))
			lstds.append(scpy.dot(self.mdp.evalftval(i), self.vst.gettLSTDr()))
			rlstds.append(scpy.dot(self.mdp.evalftval(i), self.vst.rlstd_rt))
			gtd2s.append(scpy.dot(self.mdp.evalftval(i), self.vst.gtd2_rt))
			gtd2v2s.append(scpy.dot(self.mdp.evalftval(i), self.vst.gtd2v2_rt))
			lstd0.append(scpy.dot(self.mdp.evalftval(i), self.vst.getLSTD0r()))
		self.proj_err = math.sqrt(self.mdp.norm(self.vJ-pvJ, self.std))
		self.tdlbda_err = math.sqrt(self.mdp.norm(self.vJ-tds, self.std))
		self.lstdlbda_err = math.sqrt(self.mdp.norm(self.vJ-lstds, self.std))
		self.rlstd_err = math.sqrt(self.mdp.norm(self.vJ-rlstds, self.std))
		self.gtd2_err = math.sqrt(self.mdp.norm(self.vJ-gtd2s, self.std))
		self.gtd2v2_err = math.sqrt(self.mdp.norm(self.vJ-gtd2v2s, self.std))
		self.lstd0_err = math.sqrt(self.mdp.norm(self.vJ-lstd0, self.std))
		print "Error: "
		print  [self.tdlbda_err, self.lstdlbda_err, self.rlstd_err, self.gtd2_err, self.gtd2v2_err, self.lstd0_err]
		

	def run2(self):
		global prtval
		threadLock3.acquire()
		#self.win = win
		itrcnt = 100000
		

		#clrscr(self.win, "["+str(oi)+"]", 1, 6)
		#ist = 1
		ist = scpy.random.random_integers(0, self.mdp.ssz-1)
		self.vst = ValueEstWithFuncAppr(self.mdp, ist, self.vu, self.std, self.vJ)
		thf = self.vst.getOptBasis2()
		thf.resize(len(thf)/2, 2)
		self.mdp.def_theta = thf
		#ist = np.random.randint(0,self.mdp.ssz)
		self.vst.RLSTD_init()
		prtval = 1
		ce_r = self.vst.CEMethod()
		sm = sim(self.mdp, self.vu, self.apmf, ist, itrcnt)
		for s in sm:
	#		progressbar(self.win, "[ " + str(sm.time) + " ]", 10, 6)
			writelog(1, str(sm.time) + "\n")
			self.vst.getnext(sm.time, s[0], s[1])
					
		self.evalPolicy(ce_r)
		#win.getkey()
		self.fptr.close()
		threadLock3.release()


	def run(self):
		global prtval
		threadLock3.acquire()
		#self.win = win
		itrcnt = 100000
		

		ist = scpy.random.random_integers(0, self.mdp.ssz-1)
		self.vst = ValueEstWithFuncAppr(self.mdp, ist, self.vu, self.std, self.vJ)
		thf = self.vst.getOptBasis2()
		thf.resize(len(thf)/2, 2)
		self.mdp.def_theta = thf
		#clrscr(self.win, "["+str(oi)+"]", 1, 6)
		self.actor = Actor(self.mdp, self.std)
		au = self.actor.perform()
		nu = np.array(au*self.mdp.asz, dtype=np.int)
		print "Opt.Policy: ", nu
		ce_r = self.vst.CEMethod()
		sm = sim(self.mdp, nu, self.apmf, ist, itrcnt)
		for s in sm:
	#		progressbar(self.win, "[ " + str(sm.time) + " ]", 10, 6)
			writelog(1, str(sm.time) + "\n")
			self.vst.getnext(sm.time, s[0], s[1])
					
		self.evalPolicy(ce_r)
		
		#win.getkey()
		self.fptr.close()
		threadLock3.release()

scrLock = threading.Lock()
threadLock1 = threading.Lock()
threadLock2 = threading.Lock()
threadLock3 = threading.Lock()
threads = []
#thrd1 = RL1(1,"Estimator1", vu, vJ, pu, pJ, std)
#thrd2 = RL2(2,"Estimator2", vu, vJ, pu, pJ, std)
thrd3 = RL3(3,"Estimator3", vu, vJ, pu, pJ, std)
#thrd1.start()
#thrd2.start()
thrd3.start()
#threads.append(thrd1)
#threads.append(thrd2)
threads.append(thrd3)
# Wait for all threads to complete
for trd in threads:
    trd.join()
#curses.nocbreak(); stdscr.keypad(0); curses.noecho()
#curses.endwin()
outp.printvals()

#print " "
#print OKBLUE, "Error(Function approx)", ENDC
#print "Dist(vJ, projvJ): :" , math.sqrt(dp1.norm(vJ-pvJ, std))
#print "Dist(vJ, TDvJ): :" , math.sqrt(dp1.norm(vJ-tds, std))
#print "Dist(vJ, LSTDvJ): :" , math.sqrt(dp1.norm(vJ-lstds, std))
#print "Dist(vJ, RLSTDvJ): :" , math.sqrt(dp1.norm(vJ-rlstds, std))
#print "Dist(vJ, LSPEvJ): :" , math.sqrt(dp1.norm(vJ-lspes, std))
#print "constant: ", (1.0-dp1.lbda*dp1.gamma)/(1.0-dp1.gamma)
#print " "
#print OKBLUE, "Error(Without Func approx)", ENDC
#print "Dist(vJ, TD): :" , math.sqrt(dp1.norm(vst.Jv-vJ, std))
#print "Dist(vJ, MC): :" , math.sqrt(dp1.norm(vst.TDJv-vJ, std))
#print "Dist(vJ, TD0): :" , math.sqrt(dp1.norm(vst.MCJ-vJ, std))
#tp = dp1.conv_to_policy(dp1.conv_to_decimal(pu, len(dp1.A)), len(dp1.A), len(dp1.S))
#print tp
#print np.all(pu == vu)
#print np.all(pu == tp)

		 
