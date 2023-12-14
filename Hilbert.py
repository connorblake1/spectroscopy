import numpy as np
class Hilbert:
    def __init__(self,iN,bounds,Vfxn,Pert,const_dict):
        self.hbar = const_dict['hbar']
        self.mass = const_dict['mass']
        self.c = const_dict['c']
        self.N = iN
        self.lx = bounds[0]
        self.rx = bounds[1]
        self.W = self.rx-self.lx
        self.V = np.zeros((self.N))
        self.x = np.linspace(self.lx,self.rx,iN)
        self.Vfxn = Vfxn
        self.Pert = Pert
        self.H_0 = self.buildHamiltonian(0)
        self.E0, self.Psi0 = np.linalg.eigh(self.H_0)
    def prepareC(self,coeffs):
        ncoeffs = coeffs/np.linalg.norm(coeffs)
        self.Psi = np.zeros((1,self.N))
        for i in range(self.N):
            self.Psi += ncoeffs[i]*self.Psi0[:,i]
    def prepareF(self,PsiIn,trunc):
        self.Psi = PsiIn/np.linalg.norm(PsiIn)
        ncoeffs = np.zeros((self.N,1))
        for i in range(trunc):
            ncoeffs[i] = np.dot(PsiIn,self.Psi0[:,i])
    def buildHamiltonian(self,lam):
        for i in range(self.N):
            self.V[i] = self.Vfxn(self.x[i])+self.Pert(lam,self.x[i])
        self.H = np.zeros((self.N, self.N))
        np.fill_diagonal(self.H, self.V)
        self.k = self.hbar * self.hbar / (2 * self.mass * 2*self.rx * 2*self.rx / self.N / self.N)
        for i in range(self.N):
            self.H[i, i] += 2 * self.k
            if i != self.N - 1:
                self.H[i, i + 1] += -1 * self.k
                self.H[i + 1, i] += -1 * self.k
        self.E, self.Psi = np.linalg.eigh(self.H)
        return self.H

