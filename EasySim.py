from Hilbert import Hilbert
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib
import matplotlib.colors as clrs
cms = matplotlib.cm
import os
# unperturbed Hamiltonian params
# all in hartree atomic units
c = 137 # speed of light
a = 10  # bohr radius
N = 100
m = 1  # electron rest mass
hbar = 1  # hbar
V0 = 1  # hartrees
V1 = 6
cdict = {
    "hbar":hbar,
    "mass":m,
    "c":c
}

def Vin(x): # QHO
    return V0*x*x/4
def Pert(l,x):
    #return V1*(.5*x+x*np.sin(2*x))
    #return V1*((.3*x+3.8)*(2+np.cos(x)-np.sin(x))+.08*x*x)
    #return np.sin(2.5*l)*V1*(3*np.sin(x)/x+.2*x-2)
    return np.sin(np.pi*l)*np.sin(np.pi*l)*V1*(.2*x+.006*x*x*x-.00007*x*x*x*x*x)
    #return np.power(np.sin(2*np.pi*l),2)*V1*(.2*x+np.sin(2*x)/x+.1)*3
def Vtot(l,x):
    return Vin(x) + Pert(l,x)

timespan = 10
dt = .001
tsteps = int(timespan/dt)
skipFrames = 10
pertFraction = .5
# SHOW SPECTRUM
sweep = int(tsteps/skipFrames)
lowerspec = 10
lamb = np.linspace(0,1,sweep)
Spec = np.zeros((lowerspec,sweep))
H = Hilbert(N, [-a, a],Vin,Pert, cdict)
for i,l in enumerate(lamb):
    H.buildHamiltonian(l)
    Spec[:,i] = H.E[0:lowerspec]
for i,row in enumerate(Spec):
    plt.plot(lamb,row,label=f'E{i}')
plt.legend()
plt.title("Exact Perturbed Spectrum")
plt.xlabel("Perturbation Strength")
plt.ylabel("Energy")
plt.show()
# SHOW POTENTIAL
x = np.linspace(-a,a,N)
plt.plot(x,Vtot(0,x),label="V0")
plt.plot(x,Vtot(.5,x),label="V0+$\lambda$V1")
plt.xlabel("X")
plt.ylabel("Potential")
plt.title("Perturbed Potential vs. Original Potential")
plt.legend()
plt.show()


# Animation Routine
def fullRK4(H0,H05,H10,Psii,dti):
    k1 = np.dot(H0,Psii)/1j/hbar
    k2 = np.dot(H05,Psii+k1/2*dti)/1j/hbar
    k3 = np.dot(H05,Psii+k2/2*dti)/1j/hbar
    k4 = np.dot(H10,Psii+k3*dti)/1j/hbar
    Psi_new = Psii+dti/6*(k1+2*k2+2*k3+k4)
    Psi_new /= np.linalg.norm(Psi_new)
    return Psi_new

# Build superposition of states to evolve
b = np.zeros((N,1))
b[0] = 1#/np.sqrt(2)
# b[1] = 1/2
# b[2] = 1/2
# b[3] = 1/np.sqrt(2)

Psi_0 = np.zeros((1,N))
for i in range(N):
    Psi_0 += b[i]*H.Psi0[:,i]
mixed = np.where(b != 0)[0]
print("Mixing Coeffs", mixed, b[mixed], H.E[mixed])



y_data = np.zeros((lowerspec,tsteps))
def pertL(tf): # inverted ReLu thing
    if tf < pertFraction:
        return tf/pertFraction
    else:
        return 1
# Numerical integration
Psi_evolve = np.zeros((N,tsteps),dtype=complex)
Psi_evolve[:,0] = Psi_0  # solve all timesteps for b and store in array with RK4 stepping
H10 = H.H_0
for frame in range(tsteps-1):
    H00 = H10
    H10 = H.buildHamiltonian(pertL((frame+1)/tsteps))
    H05 = .5*(H00+H10)
    print(frame)
    Psi_evolve[:,frame+1] = np.squeeze(fullRK4(H00,H05,H10,Psi_evolve[:,frame],dt))
    for eval in range(lowerspec):
        y_data[eval, frame] = np.square(np.abs(np.dot(H.Psi[:,eval],Psi_evolve[:,frame+1])))

fig = plt.figure(figsize=(5,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[8, 0],wspace=0,hspace=0)
ax = plt.subplot(gs[0])
peakAmp = np.max(np.abs(Psi_evolve))
plt.ylim(-peakAmp,peakAmp)
line1, = ax.plot(x,np.real(Psi_0.T),label="Real")  # You can change np.sin(x) to any function you want
line2, = ax.plot(x,np.imag(Psi_0.T),label="Imaginary")
norm, = ax.plot(x,np.square(np.abs(Psi_0.T)),label="Norm")
pot, = ax.plot(x,Vtot(0,x),label="Potential")
plt.title("Adiabatic Theorem")
ax.set_xlabel("X")
ax.set_ylabel("Wavefunction Amplitude")
ax.yaxis.labelpad = 1
time_label = ax.text(-a,.8*peakAmp,'',fontsize=16)
def init():
    line1.set_ydata(np.ma.array(x, mask=True))
    line2.set_ydata(np.ma.array(x, mask=True))
    norm.set_ydata(np.ma.array(x,mask=True))
    pot.set_ydata(np.ma.array(x,mask=True))
    time_label.set_text("")
    return [line1,line2, norm, pot, time_label]#+[rect for rect in b_plot]
def update(frame):
    t = frame*skipFrames/tsteps*timespan
    Psi_t = Psi_evolve[:,frame*skipFrames]
    line1.set_ydata(np.real(Psi_t.T))
    line2.set_ydata(np.imag(Psi_t.T))
    norm.set_ydata(np.square(np.abs(Psi_t.T)))
    time_label.set_text("Time: "+str(np.round(t,2)))
    pot.set_ydata(peakAmp/(Vtot(1,a))*Vtot(pertL(t/timespan),x))
    return [line1, line2, pot, norm, time_label]

ani = animation.FuncAnimation(fig, update, frames=range(int(tsteps/skipFrames)), init_func=init, blit=True)
ax.legend([line1,line2,pot,norm],["Wavefxn: Real","Wavefxn: Imaginary","Potential","Norm Square"],loc='lower right')

# SAVING
folderOut = "TDSE_Adiabatic"
perturbationType = "Cubic"
if not os.path.exists(folderOut):
    os.makedirs(folderOut)
potType = "QHO"
perturbationType = "Adiabatic"
name = ".".join(map(str,mixed))
name += "_integrated_"+str(timespan)+perturbationType+"_"+str(np.round(dt,3))+"_"+str(np.round(V1,3))
ani.save(folderOut + '\\TDSE_'+name+"_"+potType+'.gif', writer='pillow', fps=30)  # Adjust the filename and frame rate (fps) as needed
plt.cla()
plt.clf()
plt.close()

# Adiabatic Animation
fig, ax = plt.subplots()
sweep = int(tsteps/skipFrames)
lamb = np.linspace(0,1,sweep)
for i,row in enumerate(Spec):
    plt.plot(lamb,row,label=f'E{i}')
plt.legend()
plt.title("Exact Perturbed Spectrum")
plt.xlabel("Perturbation Strength")
plt.ylabel("Energy")
x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
aspect = np.abs(x_span / y_span)
rad = .07
circles = [Ellipse((0, y_data[i,0]),height=rad/aspect,width=rad,edgecolor='k',linewidth=2) for i in range(lowerspec)]
for circle in circles:
    ax.add_patch(circle)
def update2(frame):
    for i in range(lowerspec):
        if 2*frame >= int(tsteps/skipFrames)-1:
            circles[i].center = (1, Spec[i,0])
        else:
            circles[i].center = (pertL(frame*skipFrames/tsteps), Spec[i,int(frame/pertFraction)])
        circles[i].set_color(cms.jet(y_data[i,frame*skipFrames]))
    return circles
norm = clrs.Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm,ax=ax)
plt.title("Exact Perturbed Spectrum")
plt.xlabel("Perturbation Strength")
plt.ylabel("Energy")
ani2 = animation.FuncAnimation(fig, update2, frames=range(int(tsteps/skipFrames)-1), blit=True)
ani2.save(folderOut + '\\Spectrum_'+name+"_"+potType+'.gif', writer='pillow', fps=30)  # Adjust the filename and frame rate (fps) as needed

