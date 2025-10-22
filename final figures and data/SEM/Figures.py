# %% import 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
from scipy.stats import linregress
from scipy.interpolate import interp1d
import os, csv
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from matplotlib.colors import TwoSlopeNorm,SymLogNorm
from scipy.constants import c, m_e as m, hbar, e, epsilon_0 as eps0
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from scipy.interpolate import interp1d
import csv
from matplotlib.colors import SymLogNorm
from matplotlib import cm
from matplotlib.colors import Normalize

# %% functions
def k(E):
    return np.sqrt(2 * m * E) / hbar
def k_rel(E):
    return np.sqrt(E**2 + 2 * E * (m * c**2)) / (hbar * c)
def E(v0):
    """Calculate the energy of an electron with velocity v0."""
    return 0.5 * m * v0**2  # in Joules
def E_rel(v):
    """Calculate the relativistic energy of an electron with velocity v."""
    gamma = 1/np.sqrt(1 - (v**2 / c**2))
    return (gamma - 1) * m * c**2
def v_rel(E):
    """Calculate the relativistic velocity of an electron with energy E."""
    gamma = 1 +  E / (m * c**2)
    return c * np.sqrt(1 - 1/gamma**2)

def λ(E_eV):
    E = E_eV * e  # Convert eV to Joules
    return 2*np.pi * hbar*c / E
def compute_FWHM(x, y):
    half = np.max(y) / 2.0
    above = np.where(y >= half)[0]
    if len(above) < 2:
        return 0.0
    return x[above[-1]] - x[above[0]]
def FWHM(sigmaE):
    return 2*np.sqrt(2 * np.log(2)) * sigmaE
def Δk(δE_f, E0, δω, omega0, k):
    return k(E0 + δE_f + hbar*(δω)) - k(E0 + δE_f - hbar*omega0)
    return {
        'c00': c00,  # constant term
        'c10': c10,  # coefficient of δE_f
        'c01': c01,  # coefficient of δω
        'c20': c20,  # coefficient of δE_f²
        'c11': c11,  # coefficient of δE_f*δω
        'c02': c02   # coefficient of δω²
    }
def γ(gamma_dB_per_cm, v_g):
    alpha_np_per_cm = np.log(10)/10.0 * gamma_dB_per_cm   # power → Nepers (power)
    alpha_amp_per_cm = 0.5 * alpha_np_per_cm              # amplitude Nepers/cm
    alpha_np_per_m  = alpha_amp_per_cm * 100.0
    Gamma = v_g * alpha_np_per_m
    if Gamma <= 0:
        Gamma = 1e-24
    return Gamma
#  photon disperssion coefficients functions:
def q(δω,q0,vg,recoil):
    return q0 + (δω / vg) + 0.5 * recoil * δω**2
def q0_func(omega0,v0):
    E0 = E_rel(v0)                                 # central electron energy (J)
    k0 = k_rel(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    epsilon = hbar*omega0/E0
    zeta = gamma/(gamma+1)
    sigma = -1/(gamma + 1)**2
    return k0*(zeta*epsilon + sigma*epsilon**2)
def v_g_func(omega0,v0):
    E0 = E_rel(v0)                                 # central electron energy (J)
    k0 = k_rel(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    zeta = gamma/(gamma+1)
    return E0/(k0*hbar) *(1/zeta)
def recoil_func(omega0,v0):
    E0 = E_rel(v0)                                 # central electron energy (J)
    k0 = k_rel(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    sigma = -1/(gamma + 1)**2
    return  k0*hbar**2 *sigma/E0**2
def q0_CLASSICAL(omega0, v0):
    return omega0/v0 + 1/(2*k(E(v0))*(v0)**2)
def v_g_CLASSICAL(omega0, v0):
    return v0
def recoil_CLASSICAL(omega0, v0):
    return -1/((k(E(v0))*v0**2))
# numerical functions:
def nyquist_rate(v0,L_int,energy_span):
    gamma = np.sqrt(1/(1 - (v0/c)**2))
    E0 = E_rel(v0)
    k0 = k_rel(E0)
    return np.pi*E0**2 *(gamma+1)**2/(4*L_int*k0*energy_span*hbar)
# Density matrices
def final_state_probability_density(N, L_int,sigmaE_eV,v0,omega0,v_g,recoil,gamma_dB_per_cm=0):
    grid_factor = 4
    sigmaE = sigmaE_eV * e  # eV → J

    energy_span = grid_factor * sigmaE  # J
    omega_span = grid_factor * sigmaE / hbar  # rad/s


    δω = np.linspace(-omega_span, omega_span, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-energy_span, energy_span, N)  # J
    dE = δE_f[1] - δE_f[0]

    energy_span = max(abs(δE_f))
    nyquist = nyquist_rate(v0, L_int, energy_span)
    if dω > nyquist:
        print(f"Warning: δω = {dω:.3e} > Nyquist rate = {nyquist:.3e} (aliasing may occur)")

    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing="ij")
    rho_i_2d = np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * sigmaE**2)) / np.sqrt(2 * np.pi * sigmaE**2)
    i0 = np.argmin(np.abs(δω))
    K = np.zeros_like(δω)
    K[i0] = 1.0 / dω
    rho_i_e = np.sum(rho_i_2d * K[:, None], axis=0) * dω  # equals rho_i_2d[i0, :]
    rho_i_e /= np.sum(rho_i_e) * dE
    
    E0 = E_rel(v0)
    k0 = k_rel(E0)
    gamma = np.sqrt(1/(1 - (v0/c)**2))
    k0_m_hw = k_rel(E0 - hbar * omega0)
    q0 = k0 - k0_m_hw  # phase matching

    Delta_PM = (
        k_rel(E0 + δE_f_grid + hbar * δω_grid)
        - k_rel(E0 + δE_f_grid - hbar * omega0)
        - (q0 + (δω_grid / v_g) + 0.5 * recoil * δω_grid**2)
    )

    kernel = (hbar * k_rel(E0 + δE_f_grid + hbar * δω_grid) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))

    factor = e**2 * hbar * L_int**2 / (2 * eps0 * (δω_grid + omega0))
    U_factor = 1 / 4.1383282083233256e-51
    rho_f = factor * U_factor * (rho_i_2d * kernel**2)

    # Electron marginal over ω (normalized over J)
    rho_f_e = np.sum(rho_f, axis=0) * dω
    rho_f_e /= np.sum(rho_f_e * dE) if np.sum(rho_f_e * dE) != 0 else 1.0
    final_width_eV = compute_FWHM(δE_f, rho_f_e) / e

    p1 = np.sum(rho_f_e * dE)

    # Photon marginal over δE_f (normalized over rad/s)
    rho_f_p = np.sum(rho_f, axis=1) * dE
    rho_f_p /= np.sum(rho_f_p * dω) if np.sum(rho_f_p * dω) != 0 else 1.0

    # Convert grids and distributions to eV
    δE_f_eV = δE_f / e
    δω_eV = hbar * δω / e
    rho_f_e_eV = rho_f_e / e
    rho_i_e_eV = rho_i_e / e

    alpha_np_per_cm = np.log(10)/10.0 * gamma_dB_per_cm   # power → Nepers (power)
    alpha_amp_per_cm = 0.5 * alpha_np_per_cm              # amplitude Nepers/cm
    alpha_np_per_m  = alpha_amp_per_cm * 100.0
    Gamma = v_g * alpha_np_per_m
    if Gamma <= 0:
        Gamma = 1e-24
    L_eff = v_g/Gamma if Gamma > 0 else L_int
    return δE_f_eV, δω_eV, rho_f_e_eV, rho_i_e_eV, rho_f_p, final_width_eV, p1, L_eff
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
    E0 = E_rel(v0)
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
    
    for iω, ω in enumerate(δω):
        ωp = ω + u  # (M_u,)
        # Initial joint density ρ_i(E_f + ħω, E_f + ħω)
        rho_i_slice = (1 / np.sqrt(2 * np.pi * sigmaE**2)) * np.exp(
            -((δE_col + hbar * ω) ** 2) / (2 * sigmaE**2)
        )  # (1, N)

        # Phase mismatch Delta_PM (broadcasts to (M_u, N))
        Delta_PM = (
            k_rel(E0 + δE_col + hbar * ω)
            - k_rel(E0 + δE_col - hbar * omega0)
            - (q0 + (ωp[:, None] / v_g) + 0.5 * recoil * (ωp[:, None]**2))
        )

        kernel = (hbar * k_rel(E0 + δE_col + hbar * ω) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))

        factor = e**2 * hbar * L_int**2 / (2 * eps0 * (ωp[:, None] + omega0))
        

        # Lorentzian L_Γ(u) = L_Γ(ω' - ω)
        lorentz = (1 / np.pi) * (Gamma / 2.0) / ((u_col**2) + (Gamma / 2.0) ** 2)  # (M_u, 1)

        # Integrand (broadcasts lorentz to (M_u, N))
        integrand = factor *  rho_i_slice * kernel**2 * lorentz  # (M_u, N)

        # Integrate over u using Riemann sum
        integral_over_u = np.sum(integrand, axis=0) * du  # (N,)

        # Add to electron marginal (integrate over ω)
        rho_f_e += integral_over_u * dω

        # Photon marginal at this ω (integrate over E and u)
        rho_f_p[iω] = np.sum(integrand) * dE * du

    
    p1 = np.sum(rho_f_e * dE)
    
    # Normalize electron marginal
    rho_f_e /= np.sum(rho_f_e * dE) if np.sum(rho_f_e * dE) != 0 else 1.0
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
#plot functions:
def plot_v0_L_map(csv_path, initial_width, vg_fixed, E_rel, Loss):
    """
    Reads widths CSV (columns: L_int_m, v_0_m_per_s, width), builds a 2D map of
    W = log(final_width / initial_width), plots with blue=narrower, red=broader,
    and prints maximal narrowing stats incl. Energy-width ratio.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read & coerce to numeric
    df = pd.read_csv(csv_path)
    for col in ["L_int_m", "v_0_m_per_s", "width"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["L_int_m", "v_0_m_per_s", "width"])

    # Average duplicates (if file is appended over multiple runs)
    df = df.groupby(["L_int_m", "v_0_m_per_s"], as_index=False).agg({"width": "mean"})

    # Compute W = log(final / initial)
    eps = np.finfo(float).tiny
    df["W"] = np.log(np.maximum(df["width"].to_numpy(), eps) / max(initial_width, eps))

    # Build grid that auto-fits the available (L, v0) points
    L_vals  = np.sort(df["L_int_m"].unique())
    v0_vals = np.sort(df["v_0_m_per_s"].unique())
    pivot   = df.pivot(index="L_int_m", columns="v_0_m_per_s", values="W").reindex(index=L_vals, columns=v0_vals)
    Z = np.ma.masked_invalid(pivot.to_numpy())

    # --- COLOR FIX: guarantee blue only for W<0 ---
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))

    if zmin < 0 and zmax > 0:
        # Mixed signs → diverging, centered at 0
        absmax = float(np.nanmax(np.abs(Z)))
        if not np.isfinite(absmax) or absmax == 0.0:
            absmax = 1e-12
        nrm = matplotlib.colors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
        cmap = "RdBu_r"
    elif zmin >= 0:
        # All non-negative → use Reds (no blue exists)
        nrm = matplotlib.colors.Normalize(vmin=0.0, vmax=zmax if zmax > 0 else 1.0)
        cmap = "Reds"
    else:
        # All non-positive → use Blues_r
        nrm = matplotlib.colors.Normalize(vmin=zmin, vmax=0.0)
        cmap = "Blues_r"

    

    # ---- Maximal narrowing report ----
    # Find minimum final width and its (L, v0)
    # (Use the non-aggregated widths in the pivot-aligned grid)
    # Reconstruct corresponding widths_2D to match Z’s shape:
    width_pivot = (df.pivot(index="L_int_m", columns="v_0_m_per_s", values="width")
                     .reindex(index=L_vals, columns=v0_vals))
    widths_2D = width_pivot.to_numpy()
    if np.all(np.isnan(widths_2D)):
        print("No width data to summarize.")
        return

    min_idx_flat = np.nanargmin(widths_2D)
    min_i, min_j = np.unravel_index(min_idx_flat, widths_2D.shape)
    min_width = widths_2D[min_i, min_j]
    min_L_int = L_vals[min_i]
    min_v0    = v0_vals[min_j]

    # Electron kinetic energy at that v0 (nonrelativistic; use your own if needed)
    E_J  = E_rel(min_v0)  # in Joules
    E_eV = E_J / e

    print(f"Minimum width: {min_width:.6g} (same units as CSV 'width') "
          f"at L_int = {min_L_int:.6g} m, v_0 = {min_v0/c:.6g} c")
    energy_width_ratio = min_width / E_eV
    print(f"Energy-width ratio: {energy_width_ratio:.3e} (width / eV)")
    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=nrm,
        extent=[v0_vals.min(), v0_vals.max(), L_vals.min(), L_vals.max()],
        interpolation="nearest"
    )
    ax.set_xlabel(r"$v_0\ \mathrm{(m/s)}$")
    ax.set_ylabel(r"$L_{\mathrm{int}}\ \mathrm{(m)}$")
    # Title with minimal energy-width ratio and Loss [dB/cm] if Loss > 1e-3
    title = rf"Min energy-width  = {energy_width_ratio:.3e}, $v_g^\mathrm{{fixed}}={vg_fixed/c:.6g}\ \mathrm{{c}}$"
    if Loss is not None and Loss > 1e-3:
        title += rf", Loss = {Loss:.2f} dB/cm"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$W=\log(\sigma_f/\sigma_i)$")

    # Nice sci formatting on v0 axis
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.show()
def plot_v0_L_map_LOGSCALE(csv_path, initial_width, vg_fixed, E_rel,L0, Loss):
    """
    Reads widths CSV (columns: L_int_m, v_0_m_per_s, width), builds a 2D map of
    W = log(final_width / initial_width), plots with blue=narrower, red=broader,
    and prints maximal narrowing stats incl. Energy-width ratio.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read & coerce to numeric
    df = pd.read_csv(csv_path)
    for col in ["L_int_m", "v_0_m_per_s", "width"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["L_int_m", "v_0_m_per_s", "width"])

    # Average duplicates (if file is appended over multiple runs)
    df = df.groupby(["L_int_m", "v_0_m_per_s"], as_index=False).agg({"width": "mean"})

    # Compute W = log(final / initial)
    eps = np.finfo(float).tiny
    df["W"] = np.log(np.maximum(df["width"].to_numpy(), eps) / max(initial_width, eps))

    # Build grid that auto-fits the available (L, v0) points
    L_vals  = np.sort(df["L_int_m"].unique())
    v0_vals = np.sort(df["v_0_m_per_s"].unique())
    pivot   = df.pivot(index="L_int_m", columns="v_0_m_per_s", values="W").reindex(index=L_vals, columns=v0_vals)
    Z = np.ma.masked_invalid(pivot.to_numpy())

    # --- COLOR FIX: guarantee blue only for W<0 ---
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))

    if zmin < 0 and zmax > 0:
        # Mixed signs → diverging, centered at 0
        absmax = float(np.nanmax(np.abs(Z)))
        if not np.isfinite(absmax) or absmax == 0.0:
            absmax = 1e-12
        nrm = matplotlib.colors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
        cmap = "RdBu_r"
    elif zmin >= 0:
        # All non-negative → use Reds (no blue exists)
        nrm = matplotlib.colors.Normalize(vmin=0.0, vmax=zmax if zmax > 0 else 1.0)
        cmap = "Reds"
    else:
        # All non-positive → use Blues_r
        nrm = matplotlib.colors.Normalize(vmin=zmin, vmax=0.0)
        cmap = "Blues_r"

    

    # ---- Maximal narrowing report ----
    # Find minimum final width and its (L, v0)
    # (Use the non-aggregated widths in the pivot-aligned grid)
    # Reconstruct corresponding widths_2D to match Z’s shape:
    width_pivot = (df.pivot(index="L_int_m", columns="v_0_m_per_s", values="width")
                     .reindex(index=L_vals, columns=v0_vals))
    widths_2D = width_pivot.to_numpy()
    if np.all(np.isnan(widths_2D)):
        print("No width data to summarize.")
        return

    min_idx_flat = np.nanargmin(widths_2D)
    min_i, min_j = np.unravel_index(min_idx_flat, widths_2D.shape)
    min_width = widths_2D[min_i, min_j]
    min_L_int = L_vals[min_i]
    min_v0    = v0_vals[min_j]

    # Electron kinetic energy at that v0 (nonrelativistic; use your own if needed)
    E_J  = E_rel(min_v0)  # in Joules
    E_eV = E_J / e

    print(f"Minimum width: {min_width:.6g} (same units as CSV 'width') "
          f"at L_int = {min_L_int:.6g} m, v_0 = {min_v0/c:.6g} c")
    energy_width_ratio = min_width / E_eV
    print(f"Energy-width ratio: {energy_width_ratio:.3e} (width / eV)")
    L0
    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=nrm,
        extent=[v0_vals.min(), v0_vals.max(), L_vals.min(), L_vals.max()],
        interpolation="nearest"
    )
    ax.set_yscale('log')
    ax.axhline(L0, color="tab:green", linestyle="--", label=r"$L_0$")
    ax.legend()
    ax.set_xlabel(r"$v_0\ \mathrm{(m/s)}$")
    ax.set_ylabel(r"$L_{\mathrm{int}}\ \mathrm{(m)}$")
    # Title with minimal energy-width ratio and Loss [dB/cm] if Loss > 1e-3
    title = rf"Min energy-width  = {energy_width_ratio:.3e}, $v_g^\mathrm{{fixed}}={vg_fixed/c:.6g}\ \mathrm{{c}}$"
    if Loss is not None and Loss > 1e-3:
        title += rf", Loss = {Loss:.2f} dB/cm"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$W=\log(\sigma_f/\sigma_i)$")

    # Nice sci formatting on v0 axis
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.show()
def plot_v0_Lmap_COMPARISSON(file1, file2, file3, file4, initial_width, titles, figsize=(24, 5.5)):
    """
    Plots 4 v0-L maps side by side for comparison with the same color scale.

    Parameters:
    - file1, file2, file3, file4: paths to CSV files
    - initial_width: initial width value for computing W = log(final/initial)
    - titles: list of 4 titles for each subplot (optional)
    - figsize: figure size tuple
    """

    def process_data(csv_path):
        """Helper function to process a single CSV file"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Read & coerce to numeric
        df = pd.read_csv(csv_path)
        for col in ["L_int_m", "v_0_m_per_s", "width"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["L_int_m", "v_0_m_per_s", "width"])

        # Average duplicates
        df = df.groupby(["L_int_m", "v_0_m_per_s"], as_index=False).agg({"width": "mean"})

        # Compute W = log(final / initial)
        eps = np.finfo(float).tiny
        df["W"] = np.log(np.maximum(df["width"].to_numpy(), eps) / max(initial_width, eps))

        # Build grid
        L_vals = np.sort(df["L_int_m"].unique())
        v0_vals = np.sort(df["v_0_m_per_s"].unique())
        pivot = df.pivot(index="L_int_m", columns="v_0_m_per_s", values="W").reindex(index=L_vals, columns=v0_vals)
        Z = np.ma.masked_invalid(pivot.to_numpy())

        return Z, L_vals, v0_vals, df

    # Process all four datasets
    files = [file1, file2, file3, file4]
    datasets = []

    for file_path in files:
        Z, L_vals, v0_vals, df = process_data(file_path)
        datasets.append((Z, L_vals, v0_vals, df))

    # Find global min/max for consistent color scale
    all_Z_values = []
    for Z, _, _, _ in datasets:
        if not np.all(np.isnan(Z)):
            all_Z_values.extend(Z[~np.isnan(Z)].flatten())

    if not all_Z_values:
        raise ValueError("No valid data found in any of the files")

    global_zmin = np.min(all_Z_values)
    global_zmax = np.max(all_Z_values)

    # Set up normalization with global scale
    if global_zmin < 0 and global_zmax > 0:
        nrm = matplotlib.colors.TwoSlopeNorm(vmin=global_zmin, vcenter=0.0, vmax=global_zmax)
    else:
        nrm = matplotlib.colors.Normalize(vmin=global_zmin, vmax=global_zmax)

    cmap = "RdBu_r"  # Blue = narrower, Red = broader

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    # Default titles if none provided
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(4)]

    images = []

    for i, (ax, (Z, L_vals, v0_vals, df), title) in enumerate(zip(axes, datasets, titles)):
        # Plot
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=nrm,
            extent=[v0_vals.min(), v0_vals.max(), L_vals.min(), L_vals.max()],
            interpolation="nearest"
        )
        images.append(im)

        ax.set_xlabel(r"$v_0\ \mathrm{(m/s)}$")
        if i == 0:  # Only leftmost plot gets y-label
            ax.set_ylabel(r"$L_{\mathrm{int}}\ \mathrm{(m)}$")

        ax.set_title(title)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        # Print stats for each dataset
        print(f"\n--- {title} ---")
        if not np.all(np.isnan(Z)):
            min_W = np.nanmin(Z)
            max_W = np.nanmax(Z)
            print(f"W range: {min_W:.3f} to {max_W:.3f}")
        else:
            print("No valid data")

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label(r"$W=\log(\sigma_f/\sigma_i)$")

    plt.tight_layout()
    plt.subplots_adjust(right=0.93)
    plt.show()

    return fig, axes
def dispersion_plot(omega0, v0,v_g_function,recoil_function, E_function, k_function):
    # frequency grid around ω0
    omega_vec = np.linspace(0.005, 15, 200) * omega0
    δω_vec    = omega_vec - omega0

    # phase-matching linearization point (non-zero!)
    E0 = E_function(v0)                                 # central electron energy (J)
    q0_PM  = k_function(E0) - k_function(E0 - hbar*omega0)
    v_g     = v_g_function(omega0, v0)
    recoil = recoil_function(omega0, v0)

    k_diff_vec = Δk(0.0, E0, δω_vec, omega0, k_function)     # electron Δk(δω)
    q_vec      = q(δω_vec, q0_PM, v_g, recoil)           # photon q(δω)

    # Find index closest to phase-matching point (ω=ω0, k=q0_PM)
    idx_pm = np.argmin(np.abs(omega_vec - omega0))

    plt.figure()
    plt.plot(k_diff_vec, omega_vec/(2*np.pi), 'b.', label="Δk(δω)")
    plt.plot(q_vec,      omega_vec/(2*np.pi), 'r.', label="q(δω)")
    plt.scatter([q0_PM], [omega0/(2*np.pi)], color='k', marker='x', s=80, label="Phase-matching point")
    plt.xlabel("Wavenumber (1/m)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Relativistic Dispersion: Δk vs q(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()  

# %% SEM setup 
vg = 0.1 * c
v0 = vg  # electron velocity
E0 = E_rel(v0)
lambda0 = 500e-9
omega0  = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01
sigmaE =  0.1 * hbar * omega0 / e
initial_width = FWHM(sigmaE)
L0 = 1.18 * 4 * (E0 * v0 / (sigmaE * e * omega0))   # optimal interaction length
gamma_dB_per_cm = 0

recoil = recoil_func(omega0, v0)

# %%  1D graph width vs v0 :
df_v0_loaded = pd.read_csv("width_vs_v0.csv")
v0_c_vec_loaded = df_v0_loaded["v0"].to_numpy(dtype=float)/c
width_vec_loaded = df_v0_loaded["width"].to_numpy(dtype=float)

font_size = 12
plt.rcParams.update({
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
})

plt.figure(figsize=(6, 5))
plt.plot(v0_c_vec_loaded, width_vec_loaded)
plt.axhline(initial_width,color='tab:orange', linestyle=":")
plt.axvline(vg/c, color="red", linestyle="--", label=" ")
plt.xlim(v0_c_vec_loaded[0], v0_c_vec_loaded[-1])
plt.xticks(np.linspace(v0_c_vec_loaded[0], v0_c_vec_loaded[-1], 5))
plt.yticks([0, 1, 2, 3])
plt.tight_layout()


# plt.legend(frameon=False)


plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
plt.yscale("log")
plt.savefig("width_vs_v0_log.svg", format="svg")


# %% 1D graph width vs L loss and sigmaE:
L_num = 21
sigmaE_loss_simulation = 0.25 * sigmaE

L_int_vec = np.logspace(np.log10(0.5 * L0), np.log10(400 * L0), L_num)
loss_values = [0, 10, 30, 100]
loss_labels = ["0 dB/cm", "10 dB/cm", "30 dB/cm", "100 dB/cm"]
colors = ["#003366", "tab:orange", "tab:green", "tab:red"]  # dark blue, orange, green, red
cmap = plt.colormaps["Blues"]
colors = [cmap(ci) for ci in np.linspace(0.5, 1, 4)[::-1]]  # dark blue, orange, green, red

df_sigmaE_loaded = pd.read_csv("width_vs_L_for_different_sigmaE.csv")

df_loaded = pd.read_csv("width_vs_L_for_different_losses.csv")
sigmaE_factors = [0.5, 0.75, 1]
sigmaE_values = [f * sigmaE for f in sigmaE_factors]
sigmaE_labels = [rf"${f}\,\sigma_E$" for f in sigmaE_factors]
lambda_eff = 2 * np.pi * v0 / omega0
print(f"Effective wavelength: {lambda_eff*1e9:.3f} nm")

plt.figure(figsize=(8, 5))


color_indices = np.linspace(0.2, 0.85, len(sigmaE_factors))
colors_sigmaE = [cmap(ci) for ci in color_indices[::-1]]
colors_sigmaE =["tab:purple" , "tab:green", "tab:orange"]  # orange, green, purple
L0_loss = 1.18 * 4 * (E0 * v0 / (sigmaE_loss_simulation * e * omega0))   # optimal interaction length for loss sigmaE
initial_width_loss = FWHM(sigmaE_loss_simulation)
# Omit last three L_int points from all plots
omit_n = 3
for i, label in enumerate(loss_labels):
    df_plot = df_loaded[df_loaded["label"] == label]
    loss_val = df_plot["loss"].iloc[0]
    legend_label = f"{FWHM(sigmaE_loss_simulation):.3f} eV ({loss_val:.0f} dB/cm)"
    plt.plot(
        df_plot["L_int"][:-omit_n]/lambda0,
        df_plot["width"][:-omit_n] /initial_width_loss,
        color=colors[i]
    )

df_plot = df_loaded[df_loaded["label"] == "0 dB/cm"]
legend_label = f"{FWHM(sigmaE_loss_simulation):.3f} eV"
plt.plot(df_plot["L_int"][:-omit_n]/lambda0, df_plot["width"][:-omit_n] /initial_width_loss, color=colors[0],
          label=" ")

plt.axvline(L0_loss/lambda0, color=colors[0], linestyle="--", linewidth=1)

for sigmaE_val, label, color in zip(sigmaE_values, sigmaE_labels, colors_sigmaE):
    df_plot = df_sigmaE_loaded[df_sigmaE_loaded["label"] == label]
    legend_label = f"{FWHM(sigmaE_val):.3f} eV"
    plt.plot(
        df_plot["L_int"][:-omit_n]/lambda0,
        df_plot["width"][:-omit_n] /FWHM(sigmaE_val),
        label="                     ", color=color
    )
    L0_loss = 1.18 * 4 * (E0 * v0 / (sigmaE_val * e * omega0))   # optimal interaction length for loss sigmaE

    plt.axvline(L0_loss/lambda0, color=color, linestyle="--", linewidth=1)


plt.axhline(1, color="gray", linestyle=":")

plt.xlim(L_int_vec[0]/lambda0, L_int_vec[-1-omit_n]/lambda0)
plt.ylim(0.005, 4)
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()

plt.legend(frameon=False)

# plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
# plt.savefig("width_vs_L_for_different_sigmaE_and_loss.svg", format="svg")


# %% 1D graph width not normalized vs L loss and sigmaE:

L_num = 21
sigmaE_loss_simulation = 0.25 * sigmaE

L_int_vec = np.logspace(np.log10(0.5 * L0), np.log10(400 * L0), L_num)
loss_values = [0, 10, 30, 100]
loss_labels = ["0 dB/cm", "10 dB/cm", "30 dB/cm", "100 dB/cm"]
colors = ["#003366", "tab:orange", "tab:green", "tab:red"]  # dark blue, orange, green, red
cmap = plt.colormaps["Blues"]
colors = [cmap(ci) for ci in np.linspace(0.5, 1, 4)[::-1]]  # dark blue, orange, green, red

df_sigmaE_loaded = pd.read_csv("width_vs_L_for_different_sigmaE.csv")

df_loaded = pd.read_csv("width_vs_L_for_different_losses.csv")
sigmaE_factors = [0.5, 0.75, 1]
sigmaE_values = [f * sigmaE for f in sigmaE_factors]
sigmaE_labels = [rf"${f}\,\sigma_E$" for f in sigmaE_factors]
lambda_eff = 2 * np.pi * v0 / omega0
print(f"Effective wavelength: {lambda_eff*1e9:.3f} nm")

plt.figure(figsize=(8, 5))

color_indices = np.linspace(0.2, 0.85, len(sigmaE_factors))
colors_sigmaE = [cmap(ci) for ci in color_indices[::-1]]
colors_sigmaE =["tab:purple" , "tab:green", "tab:orange"]  # orange, green, purple
L0_loss = 1.18 * 4 * (E0 * v0 / (sigmaE_loss_simulation * e * omega0))   # optimal interaction length for loss sigmaE
initial_width_loss = FWHM(sigmaE_loss_simulation)
plt.axvline(L0_loss/lambda0, color=colors[0], linestyle="--", linewidth=1)
plt.axhline(initial_width_loss, color=colors[0], linestyle=":")


# Omit last three L_int points from all plots
omit_n = 3
for i, label in enumerate(loss_labels):
    df_plot = df_loaded[df_loaded["label"] == label]
    loss_val = df_plot["loss"].iloc[0]
    legend_label = f"{FWHM(sigmaE_loss_simulation):.3f} eV ({loss_val:.0f} dB/cm)"
    plt.plot(
        df_plot["L_int"][:-omit_n]/lambda0,
        df_plot["width"][:-omit_n] ,
        color=colors[i]
    )


for sigmaE_val, label, color in zip(sigmaE_values, sigmaE_labels, colors_sigmaE):
    df_plot = df_sigmaE_loaded[df_sigmaE_loaded["label"] == label]
    legend_label = f"{FWHM(sigmaE_val):.3f} eV"
    plt.plot(
        df_plot["L_int"][:-omit_n]/lambda0,
        df_plot["width"][:-omit_n] ,label=" ", color=color
    )
    L0_loss = 1.18 * 4 * (E0 * v0 / (sigmaE_val * e * omega0))   # optimal interaction length for loss sigmaE

    plt.axvline(L0_loss/lambda0, color=color, linestyle="--", linewidth=1)
    plt.axhline(FWHM(sigmaE_val), color=color, linestyle=":")


df_plot = df_loaded[df_loaded["label"] == "0 dB/cm"]
legend_label = f"{FWHM(sigmaE_loss_simulation):.3f} eV"
plt.plot(df_plot["L_int"][:-omit_n]/lambda0, df_plot["width"][:-omit_n], color=colors[0],
          label="                                ")


plt.xlim(L_int_vec[0]/lambda0, L_int_vec[-1-omit_n]/lambda0)
plt.ylim(0.003, 1.4)
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()

# plt.legend(frameon=False)

plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
plt.savefig("width_no_norm_vs_L_for_different_sigmaE_and_loss.svg", format="svg")


# %% comparison
# Load data for 0, 10, 30, 100 dB/cm
df_0db = pd.read_csv("widths_2D_v0_L_SEM_0_lin_dB_FULL_MERGED.csv")
df_10db = pd.read_csv("widths_2D_v0_L_SEM_10_lin_dB_FULL_MERGED.csv")
df_30db = pd.read_csv("widths_2D_v0_L_SEM_30_lin_dB_FULL_MERGED.csv")
df_100db = pd.read_csv("widths_2D_v0_L_SEM_100_lin_dB_FULL_MERGED.csv")

# Round to align keys (same as 10 dB plot)
for df in [df_0db, df_10db, df_30db, df_100db]:
    df["L_int_m"] = np.round(df["L_int_m"], 8)
    df["v_0_m_per_s"] = np.round(df["v_0_m_per_s"], 8)

# Get unique L_int and v0 values from df_0db (assuming consistent across files)
L_int_vec_rounded = np.sort(df_0db["L_int_m"].unique())  # 61 values
v0_vec_rounded = np.sort(df_0db["v_0_m_per_s"].unique())  # 61 values

# Process each DataFrame
widths_2D_list = []
for df in [df_0db, df_10db, df_30db, df_100db]:
    # Keep only expected coordinates
    df = df[df["L_int_m"].isin(L_int_vec_rounded)]
    df = df[df["v_0_m_per_s"].isin(v0_vec_rounded)]
    
    # Pivot with aggregation (protect against duplicates)
    pivot = df.pivot_table(index="L_int_m", columns="v_0_m_per_s", values="width", aggfunc="mean")
    pivot = pivot.reindex(index=L_int_vec_rounded, columns=v0_vec_rounded)
    
    # Sanitize and fill gaps
    W = pivot.to_numpy(dtype=float)
    bad = ~np.isfinite(W) | (W <= 0)
    W[bad] = np.nan
    dfW = pd.DataFrame(W, index=L_int_vec_rounded, columns=v0_vec_rounded)
    dfW = dfW.interpolate(axis=1, limit_direction="both")
    dfW = dfW.interpolate(axis=0, limit_direction="both")
    # CHANGE: Replaced fillna(method=...) with ffill()/bfill() to address FutureWarning
    dfW = dfW.ffill(axis=1).bfill(axis=1)
    dfW = dfW.ffill(axis=0).bfill(axis=0)
    finite_vals = dfW.to_numpy()[np.isfinite(dfW.to_numpy())]
    eps = max(1e-15, np.nanpercentile(finite_vals, 1) * 1e-6)
    W_filled = np.clip(dfW.to_numpy(), eps, None)
    widths_2D_list.append(W_filled)

widths_2D_0db, widths_2D_10db, widths_2D_30db, widths_2D_100db = widths_2D_list

# Compute log ratios
Z_0db = np.log10(widths_2D_0db/initial_width)
Z_10db = np.log10(widths_2D_10db/initial_width)
Z_30db = np.log10(widths_2D_30db/initial_width)
Z_100db = np.log10(widths_2D_100db/initial_width)

# Mask NaNs (same as original)
Z_0db_masked = np.ma.masked_invalid(Z_0db)
Z_10db_masked = np.ma.masked_invalid(Z_10db)
Z_30db_masked = np.ma.masked_invalid(Z_30db)
Z_100db_masked = np.ma.masked_invalid(Z_100db)

# Find global color scale
Wmin = float(np.nanmin([Z_0db_masked.min(), Z_10db_masked.min(), Z_30db_masked.min(), Z_100db_masked.min()]))
Wmax = float(np.nanmax([Z_0db_masked.max(), Z_10db_masked.min(), Z_30db_masked.max(), Z_100db_masked.max()]))

# Use a symmetric logarithmic normalization for the colorbar


nrm = matplotlib.colors.TwoSlopeNorm(vmin=Wmin,vcenter = 0, vmax=Wmax)
# x = max(abs(Wmax), abs(1/Wmin))
# nrm = matplotlib.colors.LogNorm(vmin=1/x, vmax=x)
cmap = plt.cm.RdBu_r

# Plotting
# Keep the plot size the same (figsize=(24, 5.5)) but increase wspace for more spacing between plots
fig, axes = plt.subplots(
    1, 4, sharey=True, figsize=(20, 5),
    gridspec_kw={"wspace": 0.01}
)

# Prepare velocity axis in units of c
v0_axis = v0_vec_rounded / c

# Plot for each loss value
for ax, Z_masked, loss in zip(
    axes,
    [Z_0db_masked, Z_10db_masked, Z_30db_masked, Z_100db_masked],
    [0, 10, 30, 100]
):
    im = ax.imshow(
        Z_masked,
        origin="lower",
        aspect="auto",
        extent=[v0_axis.min(), v0_axis.max(), L_int_vec_rounded.min()/lambda0, L_int_vec_rounded.max()/lambda0],
        cmap=cmap,
        norm=nrm
    )
    # L_th = L_threshold(initial_width, f"widths_2D_v0_L_SEM_{loss}_lin_dB.csv")
    # ax.axhline(L_th/lambda0, color="tab:purple", linestyle="--", label=r"$L_\mathrm{the}$")
    closest_L0 = L_int_vec_rounded[int(np.argmin(np.abs(L_int_vec_rounded - L0)))]
    ax.axhline(closest_L0/lambda0, color="tab:green", linestyle="--", 
               label="                ")
    ax.axvline(vg / c, color="tab:red", linestyle="--", 
               label="                               ")
    # ax.set_xlabel("Electron velocity $v_0$ (c)")
    # ax.set_title(f"Width map, loss = {loss} dB/cm")
    # ax.legend()
    # Show v0 values as plain numbers (no scientific notation)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    # Set x-ticks to a reasonable number of evenly spaced values, formatted as plain floats
    ax.set_xticks([0.09995, 0.1, 0.100025])
    ax.set_yticks([10**3, 10**4, 2*10**4])
    # Use more significant digits and avoid rounding artifacts
    # ax.set_xticklabels([f"{x:.6f}" for x in xticks])

# ax.legend()


for ax in axes:
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.savefig("widths_2D_v0_L_SEM_comparison.svg", format="svg")


# cbar = fig.colorbar(im)
# colorbar_ticks = np.array([0.04, 0.1, 1, 3])
# cbar.set_ticklabels(colorbar_ticks)
# cbar.set_ticks(np.log10(colorbar_ticks))
# cbar.set_ticklabels([])
# fig.savefig("widths_2D_colorbar_v0_L_SEM_comparison.svg", format="svg")

# %%  1D graph width vs v0 :
idx_closest = int(np.argmin(np.abs(L_int_vec_rounded - 0.01)))

width_vec_loaded = widths_2D_0db[idx_closest]/initial_width

plt.figure(figsize=(6, 5))
plt.plot(v0_axis, width_vec_loaded,'.')
plt.axhline(1,color='tab:orange', linestyle=":")
plt.axvline(vg/c, color="red", linestyle="--", label=" ")
plt.ticklabel_format(axis='x', style='plain', useOffset=False)
plt.yticks([0, 1, 2, 3])
plt.xticks([0.09995, 0.1, 0.100025])
plt.tight_layout()

# %% 2D graph: width vs v0 and initial width (sigmaE)
sigmaE_num = 81
sigmaE_values_2d = np.linspace(0.02 * sigmaE, 1.5 * sigmaE, sigmaE_num)
N = 2**10
v0_vec = v0_axis*c
widths_2D_sigmaE_v0 = np.zeros((len(sigmaE_values_2d), len(v0_vec)))

for i, sigmaE_val in enumerate(tqdm(sigmaE_values_2d)):
    for j, v0_test in enumerate(tqdm(v0_vec, desc=f"σE={sigmaE_val:.2e}", leave=False)):
        widths_2D_sigmaE_v0[i, j] = float(final_state_probability_density(
            N, L_int, sigmaE_val, v0_test, omega0,
            vg, recoil, gamma_dB_per_cm
        )[5])
# Save the 2D sigmaE-v0 width data to CSV
df_sigmaE_v0 = pd.DataFrame(
    widths_2D_sigmaE_v0,
    index=[f"{val:.6e}" for val in sigmaE_values_2d],
    columns=[f"{val:.6e}" for val in v0_vec]
)
df_sigmaE_v0.index.name = "sigmaE"
df_sigmaE_v0.columns.name = "v0"
df_sigmaE_v0.to_csv("widths_2D_sigmaE_v0_shiran.csv")


# %%  Load the 2D sigmaE-v0 width data from CSV
df_sigmaE_v0_loaded = pd.read_csv("widths_2D_sigmaE_v0_shiran.csv", index_col=0)
widths_2D_sigmaE_v0 = df_sigmaE_v0_loaded.to_numpy(dtype=float)

# Logarithmic normalization per row: white for initial width, blue < initial, red > initial
widths_2D_sigmaE_v0_lognorm = np.zeros_like(widths_2D_sigmaE_v0)
initial_widths_eV = []
for i in range(widths_2D_sigmaE_v0.shape[0]):
    initial_width_row = FWHM(sigmaE_values_2d[i]) 
    initial_widths_eV.append(initial_width_row)
    widths_2D_sigmaE_v0_lognorm[i] = np.log10(widths_2D_sigmaE_v0[i] / initial_width_row)
initial_widths_eV = np.array(initial_widths_eV)
nrm = matplotlib.colors.TwoSlopeNorm(vmin=widths_2D_sigmaE_v0_lognorm.min(),vcenter = 0, vmax=widths_2D_sigmaE_v0_lognorm.max())

cmap = "RdBu_r"
plt.figure()
plt.imshow(
    widths_2D_sigmaE_v0_lognorm,
    aspect="auto",
    origin="lower",
    extent=[v0_axis.min(), v0_axis.max(),  initial_widths_eV.min(), initial_widths_eV.max()],
    cmap=cmap,
    norm=nrm
)

plt.ticklabel_format(axis='x', style='plain', useOffset=False)
plt.xticks([0.09995, 0.1, 0.100025])
plt.yticks([0.01, 0.2, 0.4, 0.6,0.8])

plt.axvline(vg/c, color="red", linestyle="--")
# plt.legend()

cbar = plt.colorbar()
colorbar_ticks = np.array([0.04, 0.1, 1, 3])
cbar.set_ticklabels(colorbar_ticks)
cbar.set_ticks(np.log10(colorbar_ticks))
plt.tight_layout()

cbar.set_ticklabels([])
plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
plt.savefig("widths_2D_sigmaE_v0.svg", format="svg")


# %%  1D graph width vs v0 :
idx_closest = int(np.argmin(np.abs(L_int_vec_rounded - 0.01)))

width_vec_loaded_1 = widths_2D_0db[idx_closest]/initial_width

idx_closest = int(np.argmin(np.abs(initial_widths_eV - 0.584)))

width_vec_loaded_2 = widths_2D_sigmaE_v0[idx_closest]/initial_widths_eV[idx_closest]


plt.figure(figsize=(6, 5))
plt.plot(v0_axis, width_vec_loaded_1,'.')
plt.plot(v0_axis, width_vec_loaded_2,'x')
plt.axhline(1,color='tab:orange', linestyle=":")
plt.axvline(vg/c, color="red", linestyle="--", label=" ")
plt.ticklabel_format(axis='x', style='plain', useOffset=False)
plt.yticks([0, 1, 2, 3])
plt.xticks([0.09995, 0.1, 0.100025])
plt.tight_layout()


# %% 2D graph: width vs v0 and omega0
omega0_num = 41
# omega0_values_2d: logarithmically spaced from visible (light) to x-ray photon energies
# Visible: ~2.5e15 rad/s (500 nm), X-ray: ~1e19 rad/s (0.1 nm)
omega0_values_2d = np.logspace(np.log10(2.5e15), np.log10(15*omega0), omega0_num)
omega0_factors_2d = omega0_values_2d / omega0
omega0_labels_2d = [rf"${val/omega0:.2f}\,\omega_0$" for val in omega0_factors_2d]
widths_2D_omega0_v0 = np.zeros((len(omega0_values_2d), len(v0_vec)))

# N = 2**11
# for i, omega0_val in enumerate(omega0_values_2d):
#     for j, v0_test in enumerate(tqdm(v0_vec, desc=f"ω0={omega0_val:.2e}", leave=False)):
#         widths_2D_omega0_v0[i, j] = float(final_state_probability_density(
#             N, L_int, sigmaE, v0_test, omega0_val,
#             vg, recoil, gamma_dB_per_cm
#         )[5])


# # Save the 2D omega0-v0 width data to CSV after simulation
# df_omega0_v0 = pd.DataFrame(
#     widths_2D_omega0_v0,
#     index=[f"{val:.6e}" for val in omega0_values_2d],
#     columns=[f"{val:.6e}" for val in v0_vec]
# )
# df_omega0_v0.index.name = "omega0"
# df_omega0_v0.columns.name = "v0"
csv_filename = "widths_2D_omega0_v0_less_resolution.csv"
# df_omega0_v0.to_csv(csv_filename)

# --- Plot from saved CSV file ---
df_loaded = pd.read_csv(csv_filename, index_col=0)
widths_2D_omega0_v0_loaded = df_loaded.to_numpy(dtype=float)
omega0_loaded = df_loaded.index.to_numpy(dtype=float)
v0_loaded = df_loaded.columns.to_numpy(dtype=float)

lambda0_loaded = 10**9 * 2 * np.pi * c / omega0_loaded


widths_2D_omega0_v0_log = np.log10(widths_2D_omega0_v0_loaded / initial_width)
nrm = matplotlib.colors.TwoSlopeNorm(vmin=widths_2D_omega0_v0_log.min(),vcenter = 0, vmax=widths_2D_omega0_v0_log.max())

cmap = "RdBu_r"
plt.figure()
im = plt.imshow(
    widths_2D_omega0_v0_log,
    aspect="auto",
    origin="upper",  # reverse the y-axis
    extent=[v0_axis.min(), v0_axis.max(), lambda0_loaded.min(), lambda0_loaded.max()],
    cmap=cmap,
    norm=nrm
)

plt.ticklabel_format(axis='x', style='plain', useOffset=False)
plt.xticks([0.09995, 0.1, 0.100025])
# plt.yticks([10**3, 10**4, 2*10**4])

plt.axvline(vg/c, color="red", linestyle="--")
# plt.legend()

cbar = plt.colorbar()
colorbar_ticks = np.array([0.002, 0.01, 0.1, 1, 3])
cbar.set_ticklabels(colorbar_ticks)
cbar.set_ticks(np.log10(colorbar_ticks))
plt.tight_layout()

cbar.set_ticklabels([])
plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
plt.savefig("widths_2D_omega0_v0.svg", format="svg")



# %%
