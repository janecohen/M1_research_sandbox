#%% IMPORTS

import numpy as np
import matplotlib.pyplot as plt

#%% DEFINITIONS

# cavity properties
L = 1e-6 # length of the resonator
n1 = 1 # refractive index of the medium outside the cavity
n = 1.5 # refractive index of the medium inside the cavity
r = (n - n1) / (n + n1) # reflection coefficient
t = np.sqrt(1 - r**2) # transmission coefficient

c = 3e8 # speed of light
k = 2 * np.pi / (1e-6) # wave number
w = c*k # field frequency 

def Ez_inside(k, n, A, z, m):
    """
    Calculate the electric field inside the Fabry-Perot resonator.
    
    Parameters
    ----------
    k : float
        Wave number.
    n : float
        Refractive index of the medium inside the cavity.
    n1 : float
        Refractive index of the medium outside the cavity.
    A : float
        Amplitude of the incident wave.
    z : array-like
        List of positions inside the cavity where the electric field is to be calculated.
    m : int
        Mode number.

    Returns
    -------
    Ez_inside : array-like
        Electric field at the given positions z inside the cavity.
    """

    # calculate the electric field inside the cavity as a function of position z
    sign = 1 if m % 2 == 0 else -1
    E = A * np.exp(1j*k*n*z) +  sign*A * np.exp(-1j*k*n*z)
    
    return E

def Ez(k, n, n1, A, z, m):
    """
    Calculate the electric field inside and outside the Fabry-Perot resonator.
    
    Parameters
    ----------
    k : float
        Wave number.
    n : float
        Refractive index of the medium inside the cavity.
    n1 : float
        Refractive index of the medium outside the cavity.
    A : float
        Amplitude of the incident wave.
    z : array-like
        List of positions where the electric field is to be calculated.
    m : int
        Mode number.

    Returns
    -------
    Ez : array-like
        Electric field at the given positions z inside and outside the cavity.
    """

    E = np.zeros_like(z, dtype=complex)
    
    # calculate the electric field inside and outside the cavity as a function of position z
    inside = np.abs(z) <= L/2
    right = z > L/2
    left = z < -L/2

    E[inside] = Ez_inside(k, n, A, z[inside], m)

    # match the field at the cavity boundaries
    E_right = Ez_inside(k, n, A, L/2, m)
    E_left = Ez_inside(k, n, A, -L/2, m)

    # outgoing waves
    E[right] = E_right * np.exp(1j*k*n1*(z[right] - L/2))
    E[left] = E_left * np.exp(-1j*k*n1*(z[left] + L/2))
    
    return E


#%% calculate and plot the electric field for different modes

z = np.linspace(-1.5*L, 1.5*L, 2000)

for m in range(4):
    # complex QNM wave number
    k = (m*np.pi - 1j*np.log(1/r))/(n*L)

    # choose phase so even modes are cosine-like and odd modes sine-like
    A = 0.5 if m % 2 == 0 else -0.5j
    E = Ez(k, n, n1, A, z, m)
    plt.plot(z/L, np.real(E), linewidth=2, label=f"m = {m}")

plt.axvline(-0.5, color="gray", linestyle="--")
plt.axvline(0.5, color="gray", linestyle="--")
plt.xlabel(r"Position $z/L$")
plt.ylabel(r"$\mathrm{Re}[\tilde{E}_m(z)]$ (arb. units)")
plt.title("1D Fabry-Perot Cavity: Quasinormal Modes")
plt.xlim(-1.5, 1.5)
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

