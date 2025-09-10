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


# numerical functions:
def nyquist_rate(v0, L_int, energy_span):
    gamma = np.sqrt(1 / (1 - (v0 / c) ** 2))
    E0 = E_rel(v0)
    k0 = k_rel(E0)
    return np.pi * E0**2 * (gamma + 1) ** 2 / (4 * L_int * k0 * energy_span * hbar)


# %% setup:
# electron parameters
v0 = 0.1 * c  # electron velocity
sigmaE = 0.2  # eV
if v0 > 0.3 * c:
    E0 = E_rel(v0)
    k_func = k_rel
    recoil = recoil_func(v0, E0, k_func(E0))
else:
    E0 = E(v0)
    k_func = k
    recoil = recoil_CLASSICAL(v0)

# photon parameters
vg = 0.1 * c  # photon group velocity
lambda0 = 500e-9
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01  # m

sigmaE = 0.1 * hbar * omega0
initial_width = sigmaE * 2 * np.sqrt(2 * np.log(2)) / e

print(f"Initial width (FWHM): {initial_width:.4f} eV")


def final_state_probability_density(N, L_int):
    grid_factor = 4

    δω = np.linspace(-grid_factor * sigmaE / hbar, grid_factor * sigmaE / hbar, N)
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

    kernel = (hbar * k_func(E0 + δE_f_grid + hbar * δω_grid) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))

    factor = e**2 * hbar * L_int**2 / (2 * eps0 * (δω_grid + omega0))
    U_factor = 1 / 4.1383282083233256e-51
    rho_f_total = factor * U_factor * (rho_i_2d * kernel**2)
    # rho_f_total =  (rho_i_2d * kernel**2)

    # Electron marginal over ω (normalized over J)
    rho_e = np.sum(rho_f_total, axis=0) * dω
    final_width_eV = compute_FWHM(δE_f, rho_e) / e

    p1 = np.sum(rho_e * dE)
    rho_e_total = rho_e + (1 - p1) * rho_i_1d
    final_width_eV_total = compute_FWHM(δE_f, rho_e_total) / e

    # Photon marginal over δE_f (normalized over rad/s)
    rho_p = np.sum(rho_f_total, axis=1) * dE
    rho_p /= np.sum(rho_p * dω)
    return final_width_eV, final_width_eV_total, p1


def final_state_probability_density_loss(N, L_int, gamma_db_per_cm):
    grid_factor = 4

    δω = np.linspace(-grid_factor * sigmaE / hbar, grid_factor * sigmaE / hbar, N)
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

    # Losses and Lorentzian width
    alpha_np_per_cm = np.log(10.0) / 20.0 * gamma_db_per_cm
    alpha_np_per_m = alpha_np_per_cm * 100.0
    Gamma = vg * alpha_np_per_m
    if Gamma <= 0:
        Gamma = 1e-24

    # Omega bandwidth
    W = 4.0 * sigmaE / hbar

    # Local u grid for Lorentzian integration (finer and narrower)
    U = min(4.0 * Gamma, W)
    du_target = Gamma / 32.0  # Finer grid
    du = du_target if du_target > 0 else dω
    if du <= 0:
        du = dω
    M_side = max(1, int(np.ceil(U / du)))
    u = np.linspace(-M_side * du, M_side * du, 2 * M_side + 1)
    u_col = u[:, None]  # (M_u, 1)
    δE_col = δE_f[None, :]  # (1, N)

    # Initialize outputs
    rho_f = np.zeros_like(δE_f)  # Electron marginal
    # rho_f_p = np.zeros_like(δω)  # Photon marginal

    # Vectorized computation over u and E for each ω (matrix operations via broadcasting)
    for ω in tqdm(δω, desc=f"Scanning δω"):
        ωp = ω + u  # (M_u,)
        # Initial joint density ρ_i(E_f + ħω, E_f + ħω)
        rho_i_slice = (1 / np.sqrt(2 * np.pi * sigmaE**2)) * np.exp(
            -((δE_col + hbar * ω) ** 2) / (2 * sigmaE**2)
        )  # (1, N)

        # Phase mismatch Delta_PM (broadcasts to (M_u, N))
        Delta_PM = (
            k_func(E0 + δE_col + hbar * ω)
            - k_func(E0 + δE_col - hbar * omega0)
            - (q0 + (ωp[:, None] / vg) + 0.5 * recoil * (ωp[:, None] ** 2))
        )

        kernel = (hbar * k_func(E0 + δE_col + hbar * ω) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))

        factor = e**2 * hbar * L_int**2 / (2 * eps0 * (ω + omega0))
        U_factor = 1 / 4.1383282083233256e-51

        # Lorentzian L_Γ(u) = L_Γ(ω' - ω)
        lorentz = (1 / np.pi) * (Gamma / 2.0) / ((u_col**2) + (Gamma / 2.0) ** 2)  # (M_u, 1)

        # Integrand (broadcasts lorentz to (M_u, N))
        integrand = factor * U_factor * rho_i_slice * kernel**2 * lorentz  # (M_u, N)

        # Integrate over u using Riemann sum
        integral_over_u = np.sum(integrand, axis=0) * du  # (N,)

        # Add to electron marginal (integrate over ω)
        rho_f += integral_over_u * dω

        # Photon marginal at this ω (integrate over E and u)
        # rho_f_p[iω] = np.sum(integrand * dE * du)

    # Compute p1 before any normalization
    p1 = np.sum(rho_f * dE)
    print(f"p1 = {p1}")
    rho_f /= np.sum(rho_f * dE) if np.sum(rho_f * dE) != 0 else 1.0
    final_width_eV = compute_FWHM(δE_f, rho_f) / e
    # Initial 1D distribution
    rho_i_1d = (1 / np.sqrt(2 * np.pi * sigmaE**2)) * np.exp(-((δE_f) ** 2) / (2 * sigmaE**2))

    # Apply the weighted combination (no normalization of rho_f)

    rho_e_total = rho_f + (1 - p1) * rho_i_1d
    final_width_eV_total = compute_FWHM(δE_f, rho_e_total) / e

    # # Normalize photon marginal using Riemann sum
    # rho_f_p_sum = np.sum(rho_f_p * dω)
    # rho_f_p /= rho_f_p_sum if rho_f_p_sum != 0 else 1.0

    return final_width_eV, final_width_eV_total, p1


# %%
L_num = 11  # Number of interaction lengths to test
N = 2**9
L0 = (4 / np.pi) * (E0 / sigmaE) ** 2 * λ(E0 / e)  # optimal interaction length
print(f"L0 = {L0:.4f} m")
L_int_vec = np.linspace(0, 0.02, L_num)  # m
widths_L = []
probability = []
widths_L_total = []
for L_int_test in tqdm(L_int_vec, desc="Scanning L_int", position=0):
    width, width_tot, p = final_state_probability_density(N, L_int_test)
    widths_L.append(width)  # Store final width in eV
    widths_L_total.append(width_tot)  # Store total final width in eV
    probability.append(p)  # Store final probability

# %%
plt.figure()
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
L_num = 11  # Number of interaction lengths to test
N = 2**10
L0 = (4 / np.pi) * (E0 / sigmaE) ** 2 * λ(E0 / e)  # optimal interaction length
print(f"L0 = {L0:.4f} m")
L_int_vec = np.logspace(np.log10(0.0001), np.log10(0.02), L_num)  # m
widths_L = [[] for _ in range(3)]
probability = [[] for _ in range(3)]
widths_L_total = [[] for _ in range(3)]
for i, gamma_db_per_cm in enumerate([0, 30, 100]):
    for L_int_test in tqdm(L_int_vec, desc="Scanning L_int", position=0):
        width, width_tot, p = final_state_probability_density_loss(N, L_int_test, gamma_db_per_cm)
        widths_L[i].append(width)  # Store final width in eV
        widths_L_total[i].append(width_tot)  # Store total final width in eV
        probability[i].append(p)  # Store final probability
# %%
plt.figure()
for i, gamma_db_per_cm in enumerate([0, 30, 100]):
    # plt.plot(L_int_vec, np.array(widths_L[i]), ".-", label=f"Final Width {gamma_db_per_cm} dB/cm")
    plt.loglog(L_int_vec, np.array(widths_L_total[i]), ".-", label=f"Total Final Width {gamma_db_per_cm} dB/cm")


# plt.hlines(initial_width, L_int_vec[0], L_int_vec[-1], color="r", linestyle="--", label="Initial Width")
plt.loglog(L_int_vec, 1 / L_int_vec / 5000, ".-", label="1/L_int")
# plt.vlines(L0, 0, max(widths_L) * 1.1, color="g", linestyle="--", label="L0")
plt.loglog(L_int_vec, 1 / np.sqrt(L_int_vec) / 100, ".-", label="1/sqrt(L_int)")

# plt.ylim(0, initial_width * 1.1)
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length [m]")
plt.legend()
plt.show()
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
