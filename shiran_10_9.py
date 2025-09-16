# %% import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
from scipy.stats import linregress
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from scipy.constants import c, m_e as m, hbar, e, epsilon_0 as eps0



# %% functions :
# Definitions
def k(E):
    return np.sqrt(2 * m * E) / hbar


def k_rel(E):
    return np.sqrt(E**2 + 2 * E * (m * c**2)) / (hbar * c)


def E(v0):
    """Calculate the energy of an electron with velocity v0."""
    return 0.5 * m * v0**2  # in Joules


def v(E):
    """Calculate the velocity of an electron with energy E."""
    return np.sqrt(2 * E / m)  # in m/s


def E_eV(v0):
    """Calculate the energy of an electron with velocity v0."""
    return 0.5 * m * v0**2 / e  # in eV


def E_rel(v0):
    """Calculate the relativistic energy of an electron with velocity v0."""
    gamma = 1 / np.sqrt(1 - (v0**2 / c**2))
    return (gamma - 1) * m * c**2


def v_rel(E_eV):
    """Calculate the relativistic velocity of an electron with energy E."""
    E = E_eV * e  # Convert eV to Joules
    gamma = np.sqrt(1 + 2 * E / (m * c**2))
    return c * np.sqrt(1 - 1 / gamma**2)

def v_g_func(omega0,v0):
    E0 = E_rel(v0)                                 # central electron energy (J)
    k0 = k_rel(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    zeta = gamma/(gamma+1)
    return E0/(k0*hbar) *(1/zeta)

def λ(E_eV):
    E = E_eV * e  # Convert eV to Joules
    return 2 * np.pi * hbar * c / E


def compute_FWHM(x, y):
    half = np.max(y) / 2.0
    above = np.where(y >= half)[0]
    if len(above) < 2:
        return 0.0
    return x[above[-1]] - x[above[0]]


def Δk(δE_f, E0, δω, omega0, k):
    return k(E0 + δE_f + hbar * (δω)) - k(E0 + δE_f - hbar * omega0)
    return {
        "c00": c00,  # constant term
        "c10": c10,  # coefficient of δE_f
        "c01": c01,  # coefficient of δω
        "c20": c20,  # coefficient of δE_f²
        "c11": c11,  # coefficient of δE_f*δω
        "c02": c02,  # coefficient of δω²
    }


#  photon disperssion coefficients functions:
def q(δω, q0, vg, recoil):
    return q0 + (δω / vg) + 0.5 * recoil * δω**2


def q0_func(omega0, v0, E_func, k_func):
    E0 = E_func(v0)  # central electron energy (J)
    k0 = k_func(E0)  # central electron wavenumber (rad/m)
    gamma = np.sqrt(1 / (1 - (v0 / c) ** 2))
    epsilon = hbar * omega0 / E0
    zeta = gamma / (gamma + 1)
    sigma = -1 / (gamma + 1) ** 2
    return k0 * (zeta * epsilon + sigma * epsilon**2)


def recoil_func(v0, E0, k0):
    gamma = np.sqrt(1 / (1 - (v0 / c) ** 2))
    sigma = -1 / (gamma + 1) ** 2
    return k0 * hbar**2 * sigma / E0**2


def q0_CLASSICAL(omega0, v0, E_function, k_function):
    return omega0 / v0 + 1 / (2 * k(E(v0)) * (v0) ** 2)


def vg_CLASSICAL(omega0, v0, E_function, k_function):
    return v0


def recoil_CLASSICAL(v0):
    return -1 / ((k(E(v0)) * v0**2))
def final_state_probability_density_loss(N, L_int,sigmaE_eV,v0,omega0,v_g,recoil,gamma_dB_per_cm):
    grid_factor = 4
    
    sigmaE = sigmaE_eV * e 
    energy_span = grid_factor * sigmaE  # J
    omega_span = grid_factor * sigmaE / hbar
    
    δω = np.linspace(-omega_span, omega_span, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-energy_span, energy_span, N)  # J
    dE = δE_f[1] - δE_f[0]
    
    nyquist = nyquist_rate(v0, L_int, energy_span)
    if dω > nyquist:
        print(f"Warning: δω = {dω:.3e} > Nyquist rate = {nyquist:.3e} (aliasing may occur)")

    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing="ij")
    rho_i_2d = np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * sigmaE**2)) / np.sqrt(2 * np.pi * sigmaE**2)
    i0 = np.argmin(np.abs(δω))
    K = np.zeros_like(δω)
    K[i0] = 1.0 / dω
    rho_i_1d = np.sum(rho_i_2d * K[:, None], axis=0) * dω  # equals rho_i_2d[i0, :]
    rho_i_1d /= np.sum(rho_i_1d) * dE

    k0 = k_rel(E0)
    k0_m_hw = k_rel(E0 - hbar * omega0)
    q0 = k0 - k0_m_hw  # phase matching

    # Losses and Lorentzian width
    alpha_np_per_cm = np.log(10)/10.0 * gamma_dB_per_cm   # power → Nepers (power)
    alpha_amp_per_cm = 0.5 * alpha_np_per_cm              # amplitude Nepers/cm
    alpha_np_per_m  = alpha_amp_per_cm * 100.0
    Gamma = v_g * alpha_np_per_m
    if Gamma <= 0:
        Gamma = 1e-24
    L_eff = v_g/Gamma if Gamma > 0 else L_int
    # Local u grid for Lorentzian integration (finer and narrower)
    U = min(4.0 * Gamma, omega_span)
    du_target = Gamma / 32.0  # Finer grid
    du = du_target if du_target > 0 else dω
    if du <= 0:
        du = dω
    M_side = max(1, int(np.ceil(U / du)))
    u = np.linspace(-M_side * du, M_side * du, 2 * M_side + 1)
    u_col = u[:, None]  # (M_u, 1)
    δE_col = δE_f[None, :]  # (1, N)

    # Initialize outputs
    rho_f_e = np.zeros_like(δE_f)  # Electron marginal
    rho_f_p = np.zeros_like(δω)  # Photon marginal
    
    for iω, ω in enumerate(tqdm(δω, desc=f"Scanning δω")):
        ωp = ω + u  # (M_u,)
        # Initial joint density ρ_i(E_f + ħω, E_f + ħω)
        rho_i_slice = (1 / np.sqrt(2 * np.pi * sigmaE**2)) * np.exp(
            -((δE_col + hbar * ωp[:, None]) ** 2) / (2 * sigmaE**2)
        )  # (1, N)

        # Phase mismatch Delta_PM (broadcasts to (M_u, N))
        Delta_PM = (
            k_rel(E0 + δE_col + hbar * ω)
            - k_rel(E0 + δE_col - hbar * omega0)
            - (q0 + (ωp[:, None] / v_g) + 0.5 * recoil * (ωp[:, None] ** 2))
        )

        kernel = (hbar * k_rel(E0 + δE_col + hbar * ω) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))

        factor = e**2 * hbar * L_int**2 / (2 * eps0 * (ωp[:, None] + omega0))
        U_factor = 1 # / 4.1383282083233256e-51

        # Lorentzian L_Γ(u) = L_Γ(ω' - ω)
        lorentz = (1 / np.pi) * (Gamma / 2.0) / ((u_col**2) + (Gamma / 2.0) ** 2)  # (M_u, 1)

        # Integrand (broadcasts lorentz to (M_u, N))
        integrand = factor * U_factor * rho_i_slice * kernel**2 * lorentz  # (M_u, N)

        # Integrate over u using Riemann sum
        integral_over_u = np.sum(integrand, axis=0) * du  # (N,)

        # Add to electron marginal (integrate over ω)
        rho_f_e += integral_over_u * dω

        # Photon marginal at this ω (integrate over E and u)
        rho_f_p[iω] = np.sum(integrand) * dE * du

    
    p1 = np.sum(rho_f_e * dE)
    
    # Normalize electron marginal
    rho_f_e /= np.sum(rho_f_e * dE) # if np.sum(rho_f_e * dE) != 0 else 1.0
    final_width_eV = compute_FWHM(δE_f, rho_f_e) / e
    # Initial 1D distribution
    rho_i_e = (1 / np.sqrt(2 * np.pi * sigmaE**2)) * np.exp(-((δE_f) ** 2) / (2 * sigmaE**2))

    # Photon marginal over δE_f (normalized over rad/s)
    # (If you want photon marginal, sum over E for each ω)
    

    # Convert grids to eV
    δE_f_eV = δE_f / e
    δω_eV = hbar * δω / e

    # Return energy and frequency grids (in eV), final distributions, width, and p1
    return δE_f_eV, δω_eV, rho_f_e, rho_i_e, rho_f_p, final_width_eV, p1, L_eff

# numerical functions:
def nyquist_rate(v0, L_int, energy_span):
    gamma = np.sqrt(1 / (1 - (v0 / c) ** 2))
    E0 = E_rel(v0)
    k0 = k_rel(E0)
    return np.pi * E0**2 * (gamma + 1) ** 2 / (4 * L_int * k0 * energy_span * hbar)


# %% setup:




# electron parameters
sigmaE = 100e-3 *e # eV    (monochromated TEM energy spread)
E0 = 80e3  *e                   # eV    (ELECTRON ENERGY 80 keV)
v0 = v_rel(E0/e )    
# Photon & Waveguide :
λ_TEM = λ(0.8)                   # m     (wavelength) should be 0.8 - 1.2 eV
L_int = 8                   # interaction length (m)
omega0  = 2 * np.pi * c / λ_TEM               # central angular frequency (rad/s)
k_func = k_rel


N = 2**13
recoil = recoil_func(v0, E0, k_func(E0))
    

v0 = 0.1 * c  # electron velocity
E0 = E_rel(v0)         
# 
k_func = k_rel
recoil = recoil_func(v0, E0, k_func(E0))
# else:
#     E0 = E(v0)
#     k_func = k
#     recoil = recoil_CLASSICAL(v0)

vg = 0.1 * c  # photon group velocity
lambda0 = 500e-9
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
# vg = v_g_func(omega0,v0)  # photon group velocity
print(f"v0/c = {v0/c:.4f}, vg/c = {vg/c:.4f}")

sigmaE = 0.1 * hbar * omega0
initial_width = sigmaE * 2 * np.sqrt(2 * np.log(2)) / e
print(f"Initial width (FWHM): {initial_width:.6f} eV")
L0 = 0.8859/(2*np.sqrt(2*np.log(2))) * 4 * E0 * v0 / (sigmaE * (omega0 - 4 * sigmaE / hbar))  # optimal interaction length
print(f"L0 = {L0:.4f} m")
L_int = 0.0003                   # interaction length (m)

# def final_state_probability_density(N, L_int):
grid_factor = 7

δω = np.linspace(-2 * sigmaE / hbar, 2 * sigmaE / hbar, N)
dω = δω[1] - δω[0]
δE_f = np.linspace(-grid_factor * sigmaE, grid_factor * sigmaE, N)  # J
dE = δE_f[1] - δE_f[0]
#### is this correct?
energy_span = max(abs(δE_f))
####
nyquist = nyquist_rate(v0, L_int, energy_span)
if dω > nyquist:
    print(f"Warning: δω = {dω:.3e} > Nyquist rate = {nyquist:.3e} (aliasing may occur)")

δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing="ij")
rho_i_2d = np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * sigmaE**2)) / np.sqrt(2 * np.pi * sigmaE**2)
i0 = np.argmin(np.abs(δω))
K = np.zeros_like(δω)
K[i0] = 1.0 / dω
rho_i_1d = np.sum(rho_i_2d * K[:, None], axis=0) * dω  # equals rho_i_2d[i0, :]
rho_i_1d /= np.sum(rho_i_1d) * dE

k0 = k_func(E0)
k0_m_hw = k_func(E0 - hbar * omega0)
q0 = k0 - k0_m_hw  # phase matching

Delta_PM = (
    k_func(E0 + δE_f_grid + hbar * δω_grid)
    - k_func(E0 + δE_f_grid - hbar * omega0)
    - (q0 + (δω_grid / vg) + 0.5 * recoil * δω_grid**2)
)
plt.figure()
plt.imshow(Delta_PM, extent=[min(δE_f) / e, max(δE_f) / e, min(hbar * δω)/e, max(hbar * δω)/e], aspect="auto", origin="lower")
plt.colorbar(label="Delta_PM (rad/m)")
plt.xlabel("Electron Energy (eV)")
plt.ylabel("Photon Frequency (rad/s)")
plt.show()
# kernel = (hbar * k_func(E0 + δE_f_grid + hbar * δω_grid) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))
kernel = np.sinc(Delta_PM * L_int / (2 * np.pi))
plt.figure()
plt.imshow(kernel**2, extent=[min(δE_f) / e, max(δE_f) / e, min(hbar * δω)/e, max(hbar * δω)/e], aspect="auto", origin="lower")
plt.colorbar(label="|sinc|^2")
plt.xlabel("Electron Energy (eV)")
plt.ylabel("Photon Frequency (rad/s)")

plt.show()
factor = e**2 * hbar * L_int**2 / (2 * eps0 * (δω_grid + omega0))
# U_factor = 1 / 4.1383282083233256e-51
# rho_f_total = factor * U_factor * (rho_i_2d * kernel**2)
rho_f_total =  (rho_i_2d * kernel**2)

# Electron marginal over ω (normalized over J)
rho_e = np.sum(rho_f_total, axis=0) * dω
rho_e /= np.sum(rho_e * dE)
plt.figure()
plt.plot(δE_f / e, rho_i_1d, label="Initial 1D")
plt.plot(δE_f / e, rho_e, label="Final Electron")
plt.xlabel("Electron Energy (eV)")
plt.ylabel("Probability Density (1/eV)")
plt.legend()
plt.show()
final_width_eV = compute_FWHM(δE_f, rho_e) / e

# p1 = np.sum(rho_e * dE)
# rho_e_total = rho_e + (1 - p1) * rho_i_1d
# final_width_eV_total = compute_FWHM(δE_f, rho_e_total) / e

# Photon marginal over δE_f (normalized over rad/s)
rho_p = np.sum(rho_f_total, axis=1) * dE
rho_p /= np.sum(rho_p * dω)
plt.figure()
plt.plot(δω, rho_p)
plt.xlabel("Photon Frequency (rad/s)")
plt.ylabel("Probability Density (s/rad)")
plt.show()
    # return final_width_eV, final_width_eV_total, p1



# final_width_eV, final_width_eV_total, p1 = final_state_probability_density(N, L_int)
print(f"Final width (FWHM): {final_width_eV:.6f} eV")


# %% with loss

δE_f_eV, δω, rho_e, rho_e_initial, rho_f_p, final_width_eV, p1,L_eff = final_state_probability_density_loss(2**9,
                                                                                                           10,
                                                                                                            sigmaE/e,
                                                                                                            v0, omega0,
                                                                                                            vg,
                                                                                                            recoil,
                                                                                                            100)
final_state_probability_density_loss(N, L_int,sigmaE/e,v0, omega0,vg,recoil,0)[5]

plt.figure(figsize=(8, 5))
plt.plot(δE_f_eV, rho_e, label="Final electron distribution ($\\rho_f$)")
plt.plot(δE_f_eV, rho_e_initial, label="Initial electron distribution ($\\rho_i$)", linestyle="--")
plt.xlabel("Energy deviation $\\delta E_f$ (eV)")
plt.ylabel("Probability density")
plt.title(f"Electron Energy Distributions\nInitial width: {initial_width:.3g} eV, Final width: {final_width_eV:.3g} eV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot photon distribution
plt.figure(figsize=(8, 5))
plt.plot(δω, rho_f_p, label="Photon distribution")
plt.xlabel("Frequency deviation $\\delta\\omega$ (rad/s)")
plt.ylabel("Probability density")
plt.title(f"Photon Frequency Distribution\nInitial width: {initial_width:.3g} eV, Final width: {final_width_eV:.3g} eV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%% 
δE_f_eV, δω, rho_e, rho_e_initial, rho_f_p, final_width_eV, p1,L_eff = final_state_probability_density_loss(N,
                                                                                                            L_int,
                                                                                                            sigmaE,
                                                                                                            v0, omega0,
                                                                                                            vg,
                                                                                                            recoil,
                                                                                                           0)


plt.figure(figsize=(8, 5))
plt.plot(δE_f_eV, rho_e, label="Final electron distribution ($\\rho_f$)")
plt.plot(δE_f_eV, rho_e_initial, label="Initial electron distribution ($\\rho_i$)", linestyle="--")
plt.xlabel("Energy deviation $\\delta E_f$ (eV)")
plt.ylabel("Probability density")
plt.title(f"Electron Energy Distributions\nInitial width: {initial_width:.3g} eV, Final width: {final_width_eV:.3g} eV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot photon distribution
plt.figure(figsize=(8, 5))
plt.plot(δω, rho_f_p, label="Photon distribution")
plt.xlabel("Frequency deviation $\\delta\\omega$ (rad/s)")
plt.ylabel("Probability density")
plt.title(f"Photon Frequency Distribution\nInitial width: {initial_width:.3g} eV, Final width: {final_width_eV:.3g} eV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
L_num = 11  # Number of interaction lengths to test
N = 2**12
L0 = (4 / np.pi) * (E0 / sigmaE) ** 2 * λ(E0 / e)  # optimal interaction length
print(f"L0 = {L0:.4f} m")
L_int_vec = np.linspace(0, 20, L_num)  # m
widths_L = []
probability = []
widths_L_total = []
for L_int_test in tqdm(L_int_vec, desc="Scanning L_int", position=0):
    width, width_tot, p = final_state_probability_density(N, L_int_test)
    widths_L.append(width)  # Store final width in eV
    widths_L_total.append(width_tot)  # Store total final width in eV
    probability.append(p)  # Store final probability

    # Save widths_L to a file
    df = pd.DataFrame({
        "L_int": L_int_vec,
        "widths_L": widths_L,
        "widths_L_total": widths_L_total,
        "probability": probability
    })
    df.to_csv("widths_vs_L_int.csv", index=False)

# %%
plt.figure()
plt.plot(L_int_vec, np.array(widths_L), ".-", label=f"Total Final Width")

plt.plot(L_int_vec, np.array(widths_L_total), ".-", label=f"Total Final Width")
plt.hlines(initial_width, L_int_vec[0], L_int_vec[-1], color="r", linestyle="--", label="Initial Width")
plt.plot(L_int_vec, 1 / L_int_vec / 4000, ".-", label="1/L_int")
# plt.vlines(L0, 0, max(widths_L) * 1.1, color="g", linestyle="--", label="L0")
plt.plot(L_int_vec, 1 / np.sqrt(L_int_vec) / 150, ".-", label="1/sqrt(L_int)")

plt.ylim(0, initial_width * 1.1)
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length [m]")
plt.legend()
plt.show()


# %%
L_num = 21  # Number of interaction lengths to test
N = 2**12


L_int_vec = np.logspace(np.log10(0.1), np.log10(8), L_num)  # m
loss_vec = [0 , 0.5,1,2]  # dB/cm
L_eff = 1 / (np.log(10) / 20 * np.array(loss_vec) * 100)  # m
print(L_eff)

widths_L = [[] for _ in range(len(loss_vec))]
for i, gamma_db_per_cm in enumerate(loss_vec):
    for L_int_test in tqdm(L_int_vec, desc="Scanning L_int", position=0):
        res = final_state_probability_density_loss(N, L_int_test, sigmaE/e, v0, omega0, vg, recoil, gamma_db_per_cm)
        width, Leff = res[5], res[7]
        widths_L[i].append(width)  # Store final width in eV

# %%
plt.figure()
for i, gamma_db_per_cm in enumerate([0 , 0.5,1,2]):
    # plt.plot(L_int_vec, np.array(widths_L[i]), ".-", label=f"Final Width {gamma_db_per_cm} dB/cm")
    plt.loglog(L_int_vec, np.array(widths_L[i]), ".-", label=f"Final Width {gamma_db_per_cm} dB/cm")

[0, 10, 30, 100]
plt.hlines(initial_width, L_int_vec[0], L_int_vec[-1], color="r", linestyle="--", label="Initial Width")
# plt.loglog(L_int_vec[:-4], 1 / L_int_vec[:-4] / 5000, ".-", label="1/L_int")
L0 = 0.8859/(2*np.sqrt(2*np.log(2))) * 4 * E0 * v0 / (sigmaE * (omega0 - 4 * sigmaE / hbar))  # optimal interaction length
print(f"L0 = {L0:.4f} m")
plt.vlines(L0, 0, max(widths_L[0]) * 1.1, color="g", linestyle="--", label="L0")
# plt.loglog(L_int_vec[:-4], 1 / np.sqrt(L_int_vec)[:-4] / 100, ".-", label="1/sqrt(L_int)")

# plt.ylim(0, initial_width * 1.1)
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length [m]")
plt.legend()
plt.savefig("shiran_10_9_width_vs_L_int.svg", dpi=300)
# %%
plt.figure()
for i, gamma_db_per_cm in enumerate([0, 30, 100]):
    plt.plot(L_int_vec, probability[i], ".-", label=f"{gamma_db_per_cm} dB/cm")
plt.ylabel("Scattering Probability")
plt.xlabel("Interaction Length [m]")
plt.legend()
plt.show()

# %%
for i in range(3):
    print(
        (np.log10(widths_L_total[i][-1]) - np.log10(widths_L_total[i][-2]))
        / (np.log10(L_int_vec[-1]) - np.log10(L_int_vec[-2]))
    )

# %%
alpha = 7
x = np.linspace(-np.pi, np.pi, 100000)
y = np.sinc(alpha * x)**2
plt.plot(x,y,'.')
half = np.max(y) / 2.0
print(half)
above = np.where(y >= half)[0]
print(above)
comp = x[above[-1]] - x[above[0]]
FWHH = 0.8859 / alpha
print(f"FWHM = {comp}, expected {FWHH}")

integrand = np.sinc(alpha * x * dw)  **2 * 

plt.plot(x,y,'.')