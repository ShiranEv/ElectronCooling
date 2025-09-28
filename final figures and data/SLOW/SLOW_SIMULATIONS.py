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
# %% constants :
from scipy.constants import c, m_e as m, hbar, e, epsilon_0 as eps0
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from scipy.interpolate import interp1d
import csv
from matplotlib.colors import SymLogNorm
# %% functions :

# Definitions
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
    grid_factor = 5
    
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
# %%************************************************************SLOW setup************************************************************%% # 
# %% SLOW setup 
N =2**10
v0 = 0.1 * c  # electron velocity
E0 = E_rel(v0)
lambda0 = 500e-9
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01
sigmaE = 0.1 * hbar * omega0 / e
L0 = 1.18 *  4  * (E0*v0 / (sigmaE*e*omega0))   # optimal interaction length
initial_width = sigmaE * 2 * np.sqrt(2 * np.log(2)) 
gamma_dB_per_cm = 10
vg = v_g_func(omega0, v0)
recoil = recoil_func(omega0, v0)
print ("gamma value:", γ(gamma_dB_per_cm, vg))
δE_f_eV, δω, rho_e, rho_e_initial, rho_f_p, final_width_eV, p1,l_eff = final_state_probability_density(N, L_int, sigmaE, v0, omega0,
                                                                   vg, recoil, gamma_dB_per_cm)

plt.figure(figsize=(8, 5))
plt.plot(δE_f_eV, rho_e, label="Final electron distribution ($\\rho_f$)")
plt.plot(δE_f_eV, rho_e_initial, label="Initial electron distribution ($\\rho_i$)", linestyle="--")
plt.xlabel("Energy deviation $\\delta E_f$ (eV)")
plt.ylabel("Probability density")
plt.title(f"Electron Energy Distributions\nInitial width: {initial_width:.3g} eV, Final width: {final_width_eV:.3g} eV")
plt.legend()
plt.tight_layout()
plt.show()

# Plot photon distribution
plt.figure(figsize=(8, 5))
plt.plot(δω, rho_f_p, label="Photon distribution")
plt.xlabel("Frequency deviation $\\delta\\omega$ (rad/s)")
plt.ylabel("Probability density")
plt.title(f"Photon Frequency Distribution\nInitial width: {initial_width:.3g} eV, Final width: {final_width_eV:.3g} eV")
plt.legend()
plt.tight_layout()
plt.show()



# Load data for 0, 10, 30, 100 dB/cm
df_0db = pd.read_csv("widths_2D_v0_L_SEM_0_log_dB.csv")
df_10db = pd.read_csv("widths_2D_v0_L_SEM_10_log_dB.csv")
df_30db = pd.read_csv("widths_2D_v0_L_SEM_30_log_dB.csv")
df_100db = pd.read_csv("widths_2D_v0_L_SEM_100_log_dB.csv")

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
Z_0db = widths_2D_0db
Z_10db = widths_2D_10db
Z_30db = widths_2D_30db 
Z_100db = widths_2D_100db

# Mask NaNs (same as original)
Z_0db_masked = np.ma.masked_invalid(Z_0db)
Z_10db_masked = np.ma.masked_invalid(Z_10db)
Z_30db_masked = np.ma.masked_invalid(Z_30db)
Z_100db_masked = np.ma.masked_invalid(Z_100db)

# Find global color scale
Wmin = float(np.nanmin([Z_0db_masked.min(), Z_10db_masked.min(), Z_30db_masked.min(), Z_100db_masked.min()]))
Wmax = float(np.nanmax([Z_0db_masked.max(), Z_10db_masked.min(), Z_30db_masked.max(), Z_100db_masked.max()]))
linthresh = 0.05 * max(abs(Wmin), abs(Wmax))
# Use a symmetric logarithmic normalization for the colorbar

linthresh = 0.05 * max(abs(Wmin), abs(Wmax))
nrm = SymLogNorm(linthresh=linthresh, vmin=Wmin, vmax=Wmax, base=10)
cmap = "RdBu_r"

# Plotting
# CHANGE: Adjusted figure size slightly to improve spacing and prevent overlap
fig, axes = plt.subplots(1, 4, figsize=(25, 6), sharey=True)

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
        extent=[v0_axis.min(), v0_axis.max(), L_int_vec_rounded.min(), L_int_vec_rounded.max()],
        cmap=cmap,
        norm=nrm
    )
    ax.axhline(L0, color="tab:green", linestyle="--", label=r"$L_0$")
    ax.axvline(vg / c, color="tab:red", linestyle="--", label=r"$v_0 = v_g$")
    ax.set_xlabel("Electron velocity $v_0$ (c)")
    ax.set_title(f"Width map, loss = {loss} dB/cm")
    ax.legend()
    ax.ticklabel_format(axis='x', style='plain')
    ax.set_xticks(np.linspace(v0_axis.min(), v0_axis.max(), 5))

axes[0].set_ylabel("Interaction length $L_{int}$ (m)")

# CHANGE: Attach colorbar to the last subplot axis instead of using add_axes to avoid tight_layout warning
cbar = fig.colorbar(im, ax=axes[3], label="log(width / initial_width)", fraction=0.046, pad=0.04)

# CHANGE: Adjusted subplots_adjust to reserve space for colorbar, replacing tight_layout(rect=...)
fig.subplots_adjust(left=0.05, right=0.90, wspace=0.05)
plt.show()

# %% Width vs v0 
N = 2**10
v0_num = 41
vg = v_g_func(omega0, v0)
v0_vec = np.linspace(0.9999, 1.0001, v0_num) * vg  # ±1%
gamma_dB_per_cm = 0 
widths_v0_tem = []
recoil = recoil_func(omega0, v0)
for v0_test in v0_vec:
    width = final_state_probability_density(
        N, L_int, sigmaE, v0_test, omega0,
        vg, recoil, gamma_dB_per_cm
    )[5]
    widths_v0_tem.append(width)

plt.figure(figsize=(8, 5))
plt.plot(v0_vec / c, widths_v0_tem, ".-", label="Final width")
plt.axhline(initial_width, color="tab:orange", linestyle="--", label="Initial width")
plt.axvline(vg / c, color="tab:green", linestyle="--", label="$v_0 = v_g$")
plt.xlabel("Electron velocity $v_0$ (c)")
plt.ylabel("Final Width (eV)")
plt.title(f"Final Width vs. Electron Velocity (TEM setup)\n$L_0={L0:.3g}$ m")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %% 1D GRAPH: width vs L for different losses
# Load data for 0, 10, 30, 100 dB/cm
df_0db = pd.read_csv("widths_2D_v0_L_SEM_0_log_dB.csv")
df_10db = pd.read_csv("widths_2D_v0_L_SEM_10_log_dB.csv")
df_30db = pd.read_csv("widths_2D_v0_L_SEM_30_log_dB.csv")
df_100db = pd.read_csv("widths_2D_v0_L_SEM_100_log_dB.csv")

# Round to align keys (same as 10 dB plot)
for df in [df_0db, df_10db, df_30db, df_100db]:
    df["L_int_m"] = np.round(df["L_int_m"], 8)
    df["v_0_m_per_s"] = np.round(df["v_0_m_per_s"], 8)

# Get unique L_int and v0 values from df_0db (assuming consistent across files)
L_int_vec_rounded = np.sort(df_0db["L_int_m"].unique())  # 61 values
v0_vec_rounded = np.sort(df_0db["v_0_m_per_s"].unique())  # 61 values
loss_labels = [0, 10, 30, 100]
widths_2D_all = [widths_2D_0db, widths_2D_10db, widths_2D_30db, widths_2D_100db]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# Find index of v0 closest to vg
v0_idx = np.argmin(np.abs(v0_vec_rounded - vg))

plt.figure(figsize=(8, 5))
for widths, loss, color in zip(widths_2D_all, loss_labels, colors):
    width_vs_L = widths[:, v0_idx]
    plt.plot(L_int_vec_rounded, width_vs_L, marker='.', linestyle='-', label=f"{loss} dB/cm", color=color)

plt.axvline(L0, color="k", linestyle="--", label=r"$L_0$ (optimal)")
plt.axhline(initial_width, color="gray", linestyle=":", label="Initial width")
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Final width (same units as CSV)")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Width vs $L_{int}$ for $v_0 = v_g$ at different losses")
plt.legend()
plt.tight_layout()
plt.show()
# %% 1D GRAPH: width vs L for different initial widths
# 1D GRAPH: width vs L for different initial widths (sigmaE)
sigmaE_values = [sigmaE, 0.5 * sigmaE, 1.5 * sigmaE]
sigmaE_labels = [r"$\sigma_E$", r"$0.5\,\sigma_E$", r"$1.5\,\sigma_E$"]
colors_sigmaE = ["tab:blue", "tab:orange", "tab:green"]
N = 2**9
results = []

plt.figure(figsize=(8, 5))
for sigmaE_val, label, color in zip(sigmaE_values, sigmaE_labels, colors_sigmaE):
    widths_vs_L = []
    # Add tqdm progress bar for L_int_test loop
    for L_int_test in tqdm(L_int_vec_rounded, desc=f"σE={label}", leave=False):
        width = float(final_state_probability_density_loss(
            N, L_int_test, sigmaE_val, vg, omega0,
            vg, recoil, gamma_dB_per_cm
        )[5])
        widths_vs_L.append(width)
        results.append({
            "sigmaE": sigmaE_val,
            "L_int": L_int_test,
            "width": width,
            "label": label
        })
    plt.plot(L_int_vec_rounded, widths_vs_L, marker='.', linestyle='-', label=label, color=color)

# plt.axvline(L0, color="k", linestyle="--", label=r"$L_0$ (optimal)")
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Final width (eV)")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Width vs $L_{int}$ for different $\sigma_E$")
plt.legend()
plt.tight_layout()
plt.show()

# Save results to CSV
df_sigmaE = pd.DataFrame(results)
df_sigmaE.to_csv("width_vs_L_for_different_sigmaE.csv", index=False)
# %% 1D GRAPH: width vs L for different omega0

omega0_values = [omega0, 0.5 * omega0, 1.5 * omega0]
omega0_labels = [r"$\omega_0$", r"$0.5\,\omega_0$", r"$1.5\,\omega_0$"]
colors_omega0 = ["tab:blue", "tab:orange", "tab:green"]
N = 2**9
results = []

plt.figure(figsize=(8, 5))
for omega0_val, label, color in zip(omega0_values, omega0_labels, colors_omega0):
    widths_vs_L = []
    # Add tqdm progress bar for L_int_test loop
    for L_int_test in tqdm(L_int_vec_rounded, desc=f"ω0={label}", leave=False):
        width = float(final_state_probability_density_loss(
            N, L_int_test, sigmaE, vg, omega0_val,
            vg, recoil, gamma_dB_per_cm
        )[5])
        widths_vs_L.append(width)
        results.append({
            "omega0": omega0_val,
            "L_int": L_int_test,
            "width": width,
            "label": label
        })
    plt.plot(L_int_vec_rounded, widths_vs_L, marker='.', linestyle='-', label=label, color=color)

# plt.axvline(L0, color="k", linestyle="--", label=r"$L_0$ (optimal)")
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Final width (eV)")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Width vs $L_{int}$ for different $\omega_0$")
plt.legend()
plt.tight_layout()
plt.show()

# Save results to CSV
df_omega0 = pd.DataFrame(results)
df_omega0.to_csv("width_vs_L_for_different_omega0.csv", index=False)
