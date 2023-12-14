import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import os
from PIL import Image  # Import the Pillow library
def getMinFreqTime(Ei,hbari,frequencyList):
    def mean_x(factor, values):
        return sum([np.cos(2 * np.pi * v / factor) for v in values]) / len(values)
    def mean_y(factor, values):
        return sum([np.sin(2 * np.pi * v / factor) for v in values]) / len(values)
    def calculateGCDAppeal(factor, values):
        mx = mean_x(factor, values)
        my = mean_y(factor, values)
        appeal = 1 - np.sqrt(np.square((mx - 1)) + np.square(my)) / 2
        return appeal
    factor = np.linspace(Ei[0] / hbari, Ei[-1] / hbari, 1000)
    repeatFreq = -1
    appeal = -1
    for x in factor:
        val = calculateGCDAppeal(x, frequencyList)
        if val > appeal:
            appeal = val
            repeatFreq = x
    return repeatFreq
def vectorRK4(Ei,Vi,Psii,cuti,bi,ti,dti):
    bout = np.zeros_like(bi,dtype=complex)
    for n in range(cuti):
        k1 = TDSE(Ei,Vi,Psii,cuti,bi,n,ti)
        k2 = TDSE(Ei,Vi,Psii,cuti,bi+.5*k1*dti,n,ti+.5*dti)
        k3 = TDSE(Ei,Vi,Psii,cuti,bi+.5*k2*dti,n,ti+.5*dti)
        k4 = TDSE(Ei,Vi,Psii,cuti,bi+k3*dti,n,ti+dti)
        # print(k1,k2,k3,k4)
        bout[n] = bi[n] + dti*(k1+2*k2+2*k3+k4)/6
    if round(ti*1000)%100 == 0:
        print("CYCLE",np.round(ti,2),"\tNORM",np.round(np.linalg.norm(bout),10))
    bout /= np.linalg.norm(bout) # TODO
    return bout

def TDSE(Ei,Vi,Psii,cuti,bi,k,ti):
    Ni = len(Psii)
    delta = 0
    Vmat = np.zeros((Ni, Ni))
    Psi_k = Psii[:,k]
    E_k = Ei[k]
    np.fill_diagonal(Vmat,Vi)
    for n in range(cuti):
        ip = np.dot(np.conj(Psi_k),np.dot(Vmat,Psii[:,n]))
        hchange = np.exp(-1j*(Ei[n]-E_k)/hbar*ti)*bi[n]
        # print(k,n,ip,hchange)
        delta += hchange*ip
    return (delta/1j/hbar)

folderOut = "TDSE_Harmonic"
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

# unperturbed Hamiltonian params
# all in hartree atomic units
c = 137 # speed of light
a = 10  # bohr radius
N = 100
x = np.linspace(-a,a,N)
m = 1  # electron rest mass
hbar = 1  # hbar
potType = "QHO" # H0 potential for unperturbed states
V0 = 1  # hartrees
lowMix = True  # if false, does waveform

pureEvolution = False # ie only the exp(iomegat) and no time dep. pertuerbation, just stationary time evolution
# pure evolution params
minSteps = 250

# TDSE perturbation
perturbationType = "Adiabatic"  # or Harmonic
timespan = 32
dt = .001
tsteps = int(timespan/dt)
# GaussSqueeze
A0 = V0*4
window = 1.5
sigma = tsteps/window/window  # about the first windowth of time window perturbed
# Harmonic
omega = np.sqrt(2) # driving freq hbar*omega

# Build UnperterbedHamiltonian and Potential
V = np.zeros((N))
if potType == "QHO":
    for i in range(N):
        V[i] = V0*np.square(a*(i-N/2)/N)  # QHO
H_0 = np.zeros((N,N))
np.fill_diagonal(H_0,V)
k = hbar*hbar/(2*m*a*a/N/N)
for i in range(N):
    H_0[i, i] += 2*k
    if i != N-1:
        H_0[i, i+1] += -1*k
        H_0[i+1, i] += -1*k

# TODO resolve entire equation on each timestep
# Time dependent perturbation
V_t = np.zeros((N,tsteps))
center = tsteps//window
# Build TD potential
if perturbationType == "GaussSqueeze":
    for i in range(tsteps):
        # if i > center:
        #     V_t[:,i] = V*A
        # else:
        if i < center:
            ampi = A0*np.exp(-np.square((i-center)/sigma))
            V_t[:,i] = V*ampi
        else:
            V_t[:,i] = V*ampi
elif perturbationType == "Harmonic":
    for i in range(tsteps):
        V_t[:,i] = A0*np.cos(omega*i*dt)*(np.sin(4*np.pi*omega/c*x)**2)
elif perturbationType == "Adiabatic":
    V1 = np.zeros((N))
    for i in range(N):
        xi = a*(i-N/2)/(N/2)
        V1[i] = 2 * (.5 * xi + xi * np.sin(2 * xi))
    for i in range(tsteps):
        V_t[:,i] = i/tsteps*V1

# Solve
E, Psi = np.linalg.eigh(H_0)
print("k", k)
print("Energy (1-5):",E[:10])


# creation operator
A_dagger = np.zeros((N,N))
A = np.zeros_like(A_dagger)
for i in range(N-1):
    A_dagger[i+1,i] = np.sqrt(i+1)
    A[i,i+1] = np.sqrt(i+1)
A_D_Psi = np.dot(np.conj(Psi),np.dot(A_dagger,Psi))
A_Psi = np.dot(np.conj(Psi),np.dot(A,Psi))
# TODO visualize single state
# TODO commandline evolve
# TODO prep state from basis etc

# Build superposition of states to evolve
b = np.zeros((N,1))
if not lowMix:
    # wave
    waveName = "hat"
    waveblock = np.zeros((N))
    hatrange = .2
    hatsets = 2*int(N*hatrange/2)
    setrange = list(range(N//2-hatsets//2,N//2+hatsets//2))
    waveblock[setrange] = 1/np.sqrt(hatsets)
    upperCut = 15
    for i in range(upperCut):
        b[i] = np.dot(waveblock,Psi[:,i])
    b *= 1/np.linalg.norm(b) # renormalize truncated expansion
else:
    # manually define
    b[0] = 1/np.sqrt(2)
    # b[1] = 1/2
    # b[2] = 1/2
    b[3] = 1/np.sqrt(2)

print("c_norm",np.dot(b.T,b))
Psi_0 = np.zeros((1,N))
for i in range(N):
    Psi_0 += b[i]*Psi[:,i]
mixed = np.where(b != 0)[0]
print("Mixing Coeffs", mixed, b[mixed])
print("MaxFreq", max(mixed), E[max(mixed)]/hbar)
freqs = E[mixed]/hbar
repeatFreq = E[0]

# Plot and Animate
skipFrames = 50
if pureEvolution:
    minFrames = minSteps
else:
    minFrames = int(tsteps)
    b_evolve = np.zeros((N,minFrames),dtype=complex)
    b_evolve[:,0] = np.squeeze(b)  # solve all timesteps for b and store in array with RK4 stepping
    cutoff = 10  # truncate higher states, should start below cutoff/2 only
    for frame in range(minFrames-1):
        bhold = b_evolve[:,frame].copy()
        bset = vectorRK4(E,V_t[:,frame],Psi,cutoff,bhold,dt*frame,dt)
        b_evolve[:,frame+1] = np.squeeze(bset)
    # print("Bevolution")
    # for i in range(minFrames):
    #     print(b_evolve[list(range(cutoff)),i])


fig = plt.figure(figsize=(5,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1],wspace=.1,hspace=0)
ax = plt.subplot(gs[0])
peakAmp = abs(np.max(Psi[:,mixed]))
plt.ylim(-peakAmp,peakAmp)
line1, = ax.plot(x,np.real(Psi_0.T),label="Real")  # You can change np.sin(x) to any function you want
line2, = ax.plot(x,np.imag(Psi_0.T),label="Imaginary")
norm, = ax.plot(x,np.square(np.abs(Psi_0.T)),label="Norm")
pot, = ax.plot(x,V,label="Potential")
plt.title("Adiabatic Theorem")
ax.set_xlabel("X")
ax.set_ylabel("Wavefunction Amplitude")
ax.yaxis.labelpad = 1
ax_leg = plt.subplot(gs[1])
ax_leg.set_xlabel("          0 2 4 6 8 10     \nnth State",fontsize=6)
ax_leg.set_ylabel("Probability")
time_label = ax.text(-10,.22,'',fontsize=16)
cats = list(range(cutoff))
b_plot = ax_leg.bar(cats,np.zeros((cutoff)),tick_label=list(range(cutoff)))
ax_leg.set_ylim(0,1)
def init():
    line1.set_ydata(np.ma.array(x, mask=True))
    line2.set_ydata(np.ma.array(x, mask=True))
    norm.set_ydata(np.ma.array(x,mask=True))
    pot.set_ydata(np.ma.array(x,mask=True))
    time_label.set_text("")
    return [line1,line2, norm, pot, time_label]+[rect for rect in b_plot]
def update(frame):
    if pureEvolution:
        t = 2*np.pi/repeatFreq*frame/minFrames
        Psi_t = np.zeros_like(Psi[:, 0], dtype=complex)
        for i in mixed:
            Psi_t += b[i]*Psi[:,i]*np.exp(1j*E[i]/hbar*t)
        line1.set_ydata(np.real(Psi_t.T))  # You can adjust the amplitude and speed by changing the multiplier
        line2.set_ydata(np.imag(Psi_t.T))  # You can adjust the amplitude and speed by changing the multiplier
        norm.set_ydata(np.square(np.abs(Psi_t.T)))
        # print("Norm",np.square(np.abs(np.dot(np.conj(Psi_t),Psi_t))))
        time_label.set_text("Time: "+str(np.round(t,2)))
        pot.set_ydata(V)
        return line1,line2,pot,norm,time_label
    else:
        t = frame*skipFrames/minFrames*timespan
        bframe = b_evolve[:,frame*skipFrames]
        Psi_t = np.zeros_like(Psi[:, 0], dtype=complex)
        for i in range(cutoff):
            Psi_t += bframe[i]*Psi[:,i]*np.exp(1j*E[i]/hbar*t)
        line1.set_ydata(np.real(Psi_t.T))
        line2.set_ydata(np.imag(Psi_t.T))
        norm.set_ydata(np.square(np.abs(Psi_t.T)))
        # print("Norm",np.square(np.abs(np.dot(np.conj(Psi_t),Psi_t))))
        time_label.set_text("Time: "+str(np.round(t,2)))
        pot.set_ydata(V_t[:,frame*skipFrames])
        for bar,new_height in zip(b_plot,np.abs(bframe[list(range(cutoff))])):
            bar.set_height(new_height*new_height)
        return [line1,line2,pot,norm,time_label]+[rect for rect in b_plot]

print("Frames",minFrames)
ani = animation.FuncAnimation(fig, update, frames=range(int(minFrames/skipFrames)), init_func=init, blit=True)
ax.legend([line1,line2,pot,norm],["Wavefxn: Real","Wavefxn: Imaginary","Perturbation","Norm Square"],loc='lower right')

ax_leg.spines['top'].set_visible(False)
ax_leg.spines['right'].set_visible(False)
ax_leg.spines['bottom'].set_visible(False)
ax_leg.spines['left'].set_visible(False)

ax_leg.set_xticks([])
ax_leg.set_yticks([])

if len(mixed) < 4:
    name = ".".join(map(str,mixed))
else:
    name = "waveform" + waveName
if pureEvolution:
    name += "_analytic"
else:
    name += "_integrated_"+str(timespan)+perturbationType+"_"+str(np.round(A0,2))+"_"+str(np.round(dt,3))
ani.save(folderOut + '\\TDSE_'+name+"_"+potType+'.gif', writer='pillow', fps=30)  # Adjust the filename and frame rate (fps) as needed
