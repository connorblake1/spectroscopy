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



folderOut = "TDSE_Harmonic"
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

pureEvolution = True
lowMix = True # if false, does waveform

# all in hartree atomic units
a = 10  # bohr radius
N = 100
m = 1  # electron rest mass
hbar = 1  # hbar
V0 = 1  # hartrees
V_p = .3*V0
potType = "QHO"
omega = 1  # driving freq
timespan = 5
dt = .01

# Build Hamiltonian and Potential
V = np.zeros((N,1))
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

# Solve
E, Psi = np.linalg.eigh(H_0)
print("k", k)
print("Energy (1-5):",E[:5])

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
    b *= 1/np.linalg.norm(b)
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
repeatFreq = getMinFreqTime(E,hbar,freqs)
repeatFreq = min(freqs)
print("Repeat Freq",repeatFreq)
print("Repeat/Max",max(freqs)/repeatFreq)

# Plot and Animate
if pureEvolution:
    minFrames = 450
else:
    minFrames = int(timespan/dt)
fig = plt.figure(figsize=(10,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])
ax = plt.subplot(gs[0])
peakAmp = abs(np.max(Psi[:,mixed]))
plt.ylim(-peakAmp,peakAmp)
x = np.linspace(-a,a,N)
line1, = ax.plot(x,np.real(Psi_0.T),label="Real")  # You can change np.sin(x) to any function you want
line2, = ax.plot(x,np.imag(Psi_0.T),label="Imaginary")
norm, = ax.plot(x,np.square(np.abs(Psi_0.T)),label="Norm")
pot, = ax.plot(x,V,label="Potential")
def init():
    line1.set_ydata(np.ma.array(x, mask=True))
    line2.set_ydata(np.ma.array(x, mask=True))
    norm.set_ydata(np.ma.array(x,mask=True))
    pot.set_ydata(np.ma.array(x,mask=True))
    return line1,line2
def update(frame):
    if pureEvolution:
        t = 2*np.pi/repeatFreq*frame/minFrames
        Psi_t = np.zeros_like(Psi[:,0],dtype=complex)
        for i in mixed:
            Psi_t += b[i]*Psi[:,i]*np.exp(1j*E[i]/hbar*t)
        line1.set_ydata(np.real(Psi_t.T))  # You can adjust the amplitude and speed by changing the multiplier
        line2.set_ydata(np.imag(Psi_t.T))  # You can adjust the amplitude and speed by changing the multiplier
        norm.set_ydata(np.square(np.abs(Psi_t.T)))
        # print("Norm",np.square(np.abs(np.dot(np.conj(Psi_t),Psi_t))))
        pot.set_ydata(V)
        return line1,line2,pot,norm
    else:
        pass

print("Frames",minFrames)
ani = animation.FuncAnimation(fig, update, frames=range(minFrames), init_func=init, blit=True)
ax_leg = plt.subplot(gs[1])
ax_leg.legend([line1,line2,pot,norm],["Real","Imag.","Pot.","Norm"],loc='center')
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
ani.save(folderOut + '\\TDSE_'+name+"_"+potType+'.gif', writer='pillow', fps=30)  # Adjust the filename and frame rate (fps) as needed
