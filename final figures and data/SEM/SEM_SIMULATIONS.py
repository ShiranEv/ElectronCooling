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
def L_threshold(initial_width, csv_path):
    """
    Find the interaction length L where the width equals the initial width for v0 = vg.
    Args:
        initial_width (float): The initial width value to match.
        csv_path (str): Path to the CSV file with columns ["L_int_m", "v_0_m_per_s", "width"].
    Returns:
        float: The value of L_int where width crosses initial_width for v0 = vg.
    """
    df = pd.read_csv(csv_path)
    # Find unique v0 and L_int values
    v0_vec = np.sort(df["v_0_m_per_s"].unique())
    L_vec = np.sort(df["L_int_m"].unique())
    # Use global vg from context if available, otherwise use the closest v0 to the median
    try:
        v0_target = vg
    except NameError:
        v0_target = v0_vec[len(v0_vec)//2]
    # Find index of v0 closest to vg
    v0_idx = np.argmin(np.abs(v0_vec - v0_target))
    v0_closest = v0_vec[v0_idx]
    # Extract width vs L for v0 = vg
    df_vg = df[df["v_0_m_per_s"] == v0_closest]
    Ls = df_vg["L_int_m"].values
    widths = df_vg["width"].values
    # Sort by L
    sort_idx = np.argsort(Ls)
    Ls = Ls[sort_idx]
    widths = widths[sort_idx]
    # Interpolate and find L where width = initial_width
    f = interp1d(widths, Ls, kind="linear", bounds_error=False, fill_value="extrapolate")
    L_thresh = float(f(initial_width))
    return L_thresh
# %%************************************************************SEM setup************************************************************%% # 
# %% SEM setup 
N =2**10
v0 = 0.1 * c  # electron velocity
E0 = E_rel(v0)
lambda0 = 500e-9
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01
sigmaE = 0.1 * hbar * omega0 / e
L0 = 1.18 *  4  * (E0*v0 / (sigmaE*e*omega0))   # optimal interaction length
initial_width = sigmaE * 2 * np.sqrt(2 * np.log(2)) 
gamma_dB_per_cm = 0
vg = v_g_func(omega0, v0)
recoil = recoil_func(omega0, v0)
print ("gamma value:", γ(gamma_dB_per_cm, vg))
δE_f_eV, δω, rho_e, rho_e_initial, rho_f_p, final_width_eV, p1,l_eff = final_state_probability_density_loss(N, L_int, sigmaE, v0, omega0,
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

# %% 2D simulation: v0 vs L for different losses:
L_num = 41
v0_num = 41

L_int_vec = np.logspace(np.log10(0.001), np.log10(13), L_num) * L0
v0_vec = np.logspace(np.log10(0.9995), np.log10(1.0005), v0_num) * vg
# v0_vec = np.linspace(0.9995, 1.0005, v0_num) * vg
widths_2D_tem = np.zeros((len(L_int_vec), len(v0_vec)))
gamma_dB_per_cm = 30
N = 2**10
L0 = 1.18 * 4 * (E0 * v0 / (sigmaE * e * omega0))   # optimal interaction length
vg = v_g_func(omega0, v0)
recoil = recoil_func(omega0, v0)
_rows_tem = []
for i, L_int_test in enumerate(tqdm(L_int_vec, desc="Scanning L_int (TEM)", position=0)):
    print(f"Scanning L_int = {L_int_test:.5f} m")
    for j, v0_test in enumerate(tqdm(v0_vec, desc=f"Scanning v_0 for L_int={L_int_test:.5f}", leave=False, position=1)):
        width = float(final_state_probability_density_loss(
            N, L_int_test, sigmaE, v0_test, omega0,
            vg, recoil, gamma_dB_per_cm
        )[5])
        widths_2D_tem[i, j] = width
        _rows_tem.append((float(L_int_test), float(v0_test), width))
# Save to CSV
ACCUM_CSV_SEM = "widths_2D_v0_L_SEM_30_log_dB.csv"
df_tem = pd.DataFrame(_rows_tem, columns=["L_int_m", "v_0_m_per_s", "width"])
df_tem.to_csv(ACCUM_CSV_SEM, index=False)
L_threshold = L_int_vec[np.argmin(np.abs(widths_2D_tem[:, int(np.floor(v0_num/2))] - initial_width))]

# %% 
# Load
df_loaded = pd.read_csv("widths_2D_v0_L_SEM_0_lin_dB.csv")

# Round to align keys (same rounding as vectors)
L_int_vec_rounded = np.sort(df_loaded["L_int_m"].unique())
v0_vec_rounded = np.sort(df_loaded["v_0_m_per_s"].unique())

# Keep only expected coordinates
df_loaded = df_loaded[df_loaded["L_int_m"].isin(L_int_vec_rounded)]
df_loaded = df_loaded[df_loaded["v_0_m_per_s"].isin(v0_vec_rounded)]

# Pivot with aggregation (protect against accidental duplicates)
pivot = df_loaded.pivot_table(index="L_int_m",
                              columns="v_0_m_per_s",
                              values="width",
                              aggfunc="mean")

# Reindex to full grid order
pivot = pivot.reindex(index=L_int_vec_rounded, columns=v0_vec_rounded)

# ---- Only mask invalid data, no interpolation ----
W = pivot.to_numpy(dtype=float)
bad = ~np.isfinite(W) | (W <= 0)
W[bad] = np.nan

# 4) final safety floor to avoid log(0)
finite_vals = W[np.isfinite(W)]
eps = max(1e-15, np.nanpercentile(finite_vals, 1) * 1e-6)
W = np.clip(W, eps, None)

# Set color normalization so that white means initial width, log scale
vmin = np.nanmin(W)
vmax = np.nanmax(W)
linthresh = 0.05 * max(abs(vmax - initial_width), abs(initial_width - vmin))
nrm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
cmap = "RdBu_r"

L_th = L_threshold(initial_width, "widths_2D_v0_L_SEM_0_lin_dB.csv")
plt.figure(figsize=(8, 6))
plt.title("Width vs L_int and v0 (SEM setup), loss = 0 dB/cm")

im = plt.imshow(
    W,
    origin="lower",
    aspect="auto",
    extent=[v0_vec_rounded.min()/c, v0_vec_rounded.max()/c,
            L_int_vec_rounded.min(), L_int_vec_rounded.max()],
    cmap=cmap,
    norm=nrm,
)
# Set colorbar with white at initial_width
cbar = plt.colorbar(im, label=r"Final width (white = initial width)")
cbar.set_ticks([vmin, initial_width, vmax])
cbar.ax.set_yticklabels([f"{vmin:.2g}", f"{initial_width:.2g}", f"{vmax:.2g}"])

plt.axhline(L0, color="tab:green", linestyle="--", label=r"$L_0$")
plt.axhline(L_th, color="tab:purple", linestyle="--", label=r"$L_\mathrm{the}$")
#plt.axvline(vg / c, color="tab:red", linestyle="--", label=r"$v_0 = v_g$")
plt.xlabel("Electron velocity $v_0$ (c)")
plt.ylabel("Interaction length $L_{int}$ (m)")
plt.title("Width vs L_int and v0 (SEM setup), loss = 0 dB/cm")
plt.legend()
plt.tight_layout()
plt.show()
# %% comparison
# Load data for 0, 10, 30, 100 dB/cm
df_0db = pd.read_csv("widths_2D_v0_L_SEM_0_lin_dB.csv")
df_10db = pd.read_csv("widths_2D_v0_L_SEM_10_lin_dB.csv")
df_30db = pd.read_csv("widths_2D_v0_L_SEM_30_lin_dB.csv")
df_100db = pd.read_csv("widths_2D_v0_L_SEM_100_lin_dB.csv")

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

linthresh = 0.02 * max(abs(vmax - initial_width), abs(initial_width - vmin))
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
    # L_th = L_threshold(initial_width, f"widths_2D_v0_L_SEM_{loss}_lin_dB.csv")
    ax.axhline(L_th, color="tab:purple", linestyle="--", label=r"$L_\mathrm{the}$")
    ax.axhline(L0, color="tab:green", linestyle="--", label=r"$L_0$")
    ax.axvline(vg / c, color="tab:red", linestyle="--", label=r"$v_0 = v_g$")
    ax.set_xlabel("Electron velocity $v_0$ (c)")
    ax.set_title(f"Width map, loss = {loss} dB/cm")
    ax.legend()
    # Show v0 values as plain numbers (no scientific notation)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    # Set x-ticks to a reasonable number of evenly spaced values, formatted as plain floats
    xticks = np.linspace(v0_axis.min(), v0_axis.max(), 5)
    ax.set_xticks(xticks)
    # Use more significant digits and avoid rounding artifacts
    ax.set_xticklabels([f"{x:.6f}" for x in xticks])

axes[0].set_ylabel("Interaction length $L_{int}$ (m)")

# CHANGE: Attach colorbar to the last subplot axis instead of using add_axes to avoid tight_layout warning
cbar = fig.colorbar(im, ax=axes[3], label="log(width / initial_width)", fraction=0.046, pad=0.04)

# CHANGE: Adjusted subplots_adjust to reserve space for colorbar, replacing tight_layout(rect=...)
fig.subplots_adjust(left=0.05, right=0.90, wspace=0.05)
plt.show()
# Save the last 2D comparison figure as SVG
# fig.savefig("widths_2D_v0_L_SEM_comparison.svg", format="svg")

# %% 1D GRAPH: width vs L for different losses
L_num = 11
L_int_vec = np.logspace(np.log10(0.001* L0), np.log10(40* L0), L_num) 
loss_values = [0, 10, 30, 100]
loss_labels = ["0 dB/cm", "10 dB/cm", "30 dB/cm", "100 dB/cm"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
N = 2**11
# Run simulation and save results to CSV
results = []
for gamma_dB_per_cm, label in zip(loss_values, loss_labels):
    for L_int_test in tqdm(L_int_vec, desc=f"Loss={gamma_dB_per_cm} dB/cm", leave=False):
        width = float(final_state_probability_density_loss(
            N, L_int_test, sigmaE, v0, omega0,
            vg, recoil, gamma_dB_per_cm
        )[5])
        results.append({
            "loss": gamma_dB_per_cm,
            "label": label,
            "L_int": L_int_test,
            "width": width
        })

df_widths_vs_L_loss = pd.DataFrame(results)
df_widths_vs_L_loss.to_csv("width_vs_L_for_different_losses.csv", index=False)

# Load results from CSV and plot
df_loaded = pd.read_csv("width_vs_L_for_different_losses.csv")


plt.figure(figsize=(8, 5))
for label, color in zip(loss_labels, colors):
    df_plot = df_loaded[df_loaded["label"] == label]
    plt.plot(df_plot["L_int"], df_plot["width"], marker='.', linestyle='-', label=label, color=color)

plt.axvline(L0, color="k", linestyle="--", label=r"$L_0$ (optimal)")
plt.axhline(initial_width, color="gray", linestyle=":", label="Initial width")
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Final width (same units as CSV)")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Width vs $L_{int}$ for $v_0 = v_g$ at different losses")
plt.legend()
plt.tight_layout()
plt.savefig("width_vs_L_for_different_losses.svg", format="svg")
plt.show()

# %% 1D GRAPH: width vs L for different initial widths
L_int_vec = np.logspace(np.log10(0.001* L0), np.log10(40* L0), L_num) 
sigmaE_factors = [0.1, 0.25, 0.5, 1.5]
sigmaE_values = [f * sigmaE for f in sigmaE_factors]
sigmaE_labels = [rf"${f}\,\sigma_E$" for f in sigmaE_factors]
N = 2**11
results_sigmaE = []
gamma_dB_per_cm = 0 
for sigmaE_val, label in zip(sigmaE_values, sigmaE_labels):
    widths_vs_L = []
    for L_int_test in tqdm(L_int_vec, desc=f"σE={label}", leave=False):
        width = float(final_state_probability_density(
            N, L_int_test, sigmaE_val, v0, omega0,
            vg, recoil, gamma_dB_per_cm
        )[5])
        widths_vs_L.append(width)
        results_sigmaE.append({
            "sigmaE": sigmaE_val,
            "L_int": L_int_test,
            "width": width,
            "label": label
        })

# Save results to CSV
df_sigmaE = pd.DataFrame(results_sigmaE)
df_sigmaE.to_csv("width_vs_L_for_different_sigmaE.csv", index=False)

# Step 2: Load results from CSV and plot
df_sigmaE_loaded = pd.read_csv("width_vs_L_for_different_sigmaE.csv")
plt.figure(figsize=(8, 5))
markers = ['o', '*', '^', 'D']
for label, marker in zip(sigmaE_labels, markers):
    df_plot = df_sigmaE_loaded[df_sigmaE_loaded["label"] == label]
    plt.plot(df_plot["L_int"], df_plot["width"], marker=marker, linestyle='-', label=label, markersize=5)
plt.axvline(L0, color="k", linestyle="--", label=r"$L_0$ (optimal)")
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Final width (eV)")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Width vs $L_{int}$ for different initial widths $\sigma_E$")
plt.legend()
plt.tight_layout()
plt.savefig("width_vs_L_for_different_sigmaE.svg", format="svg")
plt.show()
# %% 1D GRAPH: width vs L for different omega0
# Create L_int_vec and v0_vec in logscale
# L_int_vec: logarithmically spaced from 0.001*L0 to 15*L0, 30 points
L_int_vec = np.logspace(np.log10(0.001 * L0), np.log10(15 * L0), 30)

# Add more omega0 values for a broader sweep
omega0_factors = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
omega0_values = [f * omega0 for f in omega0_factors]
omega0_labels = [rf"${f}\,\omega_0$" if f != 1.0 else r"$\omega_0$" for f in omega0_factors]
colors_omega0 = ["tab:purple", "tab:blue", "tab:cyan", "tab:orange", "tab:green", "tab:red", "tab:brown"]
N = 2**9
results = []

# Run simulation and save results to CSV
for omega0_val, label, color in zip(omega0_values, omega0_labels, colors_omega0):
    widths_vs_L = []
    for L_int_test in tqdm(L_int_vec, desc=f"ω0={label}", leave=False):
        width = float(final_state_probability_density(
            N, L_int_test, sigmaE, v0, omega0_val,
            vg, recoil, gamma_dB_per_cm
        )[5])
        widths_vs_L.append(width)
        results.append({
            "omega0": omega0_val,
            "L_int": L_int_test,
            "width": width,
            "label": label
        })
# Save results to CSV
df_omega0 = pd.DataFrame(results)
df_omega0.to_csv("width_vs_L_for_different_omega0.csv", index=False)

# Load results from CSV and plot
df_omega0_loaded = pd.read_csv("width_vs_L_for_different_omega0.csv")
plt.figure(figsize=(9, 6))
for label, color in zip(omega0_labels, colors_omega0):
    df_plot = df_omega0_loaded[df_omega0_loaded["label"] == label]
    plt.plot(df_plot["L_int"], df_plot["width"], marker='.', linestyle='-', label=label, color=color)
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Final width (eV)")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Width vs $L_{int}$ for different $\omega_0$")
plt.legend()
plt.tight_layout()
plt.show()
# Save the last figure as SVG
plt.savefig("width_vs_L_for_different_omega0.svg", format="svg")

# %% 1D graph width vs v0 :
v0_num = 41

v0_vec = np.linspace(0.999, 1.001, v0_num) * vg
widths_1D_v0 = np.zeros(len(v0_vec))        
N = 2**9
for i, v0_test in enumerate(tqdm(v0_vec, desc="Scanning v_0", position=0)):
    widths_1D_v0[i] = float(final_state_probability_density(
        N, L_int, sigmaE, v0_test, omega0,
        vg, recoil, gamma_dB_per_cm
    )[5])   
plt.figure(figsize=(8, 5))
plt.plot(v0_vec/c, widths_1D_v0, marker='.', linestyle='-')
plt.axhline(initial_width, color="gray", linestyle=":", label="Initial width")  
plt.axvline(vg/c, color="red", linestyle="--", label=r"$v_0 = v_g$")
plt.xlabel("Electron velocity $v_0$ (c)")
plt.ylabel("Final width (eV)")
plt.title("Final width vs $v_0$\n($L_{int} = %.3g$ m)" % L_int)
plt.legend()
plt.tight_layout()
plt.show()

# %% 2D graph: width vs v0 and initial width (sigmaE)
sigmaE_num = 41
v0_num = 41
v0_vec = np.linspace(0.999, 1.001, v0_num) * vg
sigmaE_values_2d = np.linspace(0.005 * sigmaE, 2.5 * sigmaE, sigmaE_num)
sigmaE_labels_2d = [rf"${val/sigmaE:.2f}\,\sigma_E$" for val in sigmaE_values_2d]
widths_2D_sigmaE_v0 = np.zeros((len(sigmaE_values_2d), len(v0_vec)))

N = 2**8
for i, sigmaE_val in enumerate(sigmaE_values_2d):
    for j, v0_test in enumerate(tqdm(v0_vec, desc=f"σE={sigmaE_val:.2e}", leave=False)):
        widths_2D_sigmaE_v0[i, j] = float(final_state_probability_density(
            N, L_int, sigmaE_val, v0_test, omega0,
            vg, recoil, gamma_dB_per_cm
        )[5])

# Use log red-blue coloring (SymLogNorm) as before
finite_vals = widths_2D_sigmaE_v0[np.isfinite(widths_2D_sigmaE_v0)]
Wmin = float(np.nanmin(finite_vals))
Wmax = float(np.nanmax(finite_vals))
linthresh = 0.08 * max(abs(Wmin), abs(Wmax))
nrm = SymLogNorm(linthresh=linthresh, vmin=Wmin, vmax=Wmax, base=10)
cmap = "RdBu_r"
plt.figure(figsize=(8, 5))
im = plt.imshow(
    widths_2D_sigmaE_v0,
    aspect="auto",
    origin="lower",
    extent=[v0_vec.min()/c, v0_vec.max()/c, 0, len(sigmaE_values_2d)-1],
    cmap=cmap,
    norm=nrm
)
# Reduce number of yticks and xticks for clarity
num_yticks = 6
num_xticks = 6
ytick_indices = np.linspace(0, len(sigmaE_values_2d)-1, num_yticks, dtype=int)
xtick_vals = np.linspace(v0_vec.min()/c, v0_vec.max()/c, num_xticks)
plt.yticks(ticks=ytick_indices, labels=[sigmaE_labels_2d[i] for i in ytick_indices])
plt.xticks(ticks=xtick_vals, labels=[f"{x:.4f}" for x in xtick_vals])
plt.xlabel("Electron velocity $v_0$ (c)")
plt.ylabel("Initial width $\sigma_E$")
plt.title("Final width vs $v_0$ and initial width $\sigma_E$\n($L_{int} = %.3g$ m)" % L_int)
plt.axvline(vg/c, color="red", linestyle="--", label=r"$v_0 = v_g$")
plt.legend()
cbar = plt.colorbar(im, label="Final width (eV)")
plt.tight_layout()
plt.show()

# Save the 2D sigmaE-v0 width data to CSV
df_sigmaE_v0 = pd.DataFrame(
    widths_2D_sigmaE_v0,
    index=[f"{val:.6e}" for val in sigmaE_values_2d],
    columns=[f"{val:.6e}" for val in v0_vec]
)
df_sigmaE_v0.index.name = "sigmaE"
df_sigmaE_v0.columns.name = "v0"
df_sigmaE_v0.to_csv("widths_2D_sigmaE_v0(3rd_more_resolution).csv")
# Save the last 2D sigmaE-v0 width figure as SVG
plt.savefig("widths_2D_sigmaE_v0.svg", format="svg")

# %% 2D graph: width vs v0 and omega0
omega0_num = 41
v0_num = 41
v0_vec = np.linspace(0.99, 1.01, v0_num) * vg

# omega0_values_2d: logarithmically spaced from visible (light) to x-ray photon energies
# Visible: ~2.5e15 rad/s (500 nm), X-ray: ~1e19 rad/s (0.1 nm)
omega0_values_2d = np.logspace(np.log10(2.5e15), np.log10(1e18), omega0_num)
omega0_factors_2d = omega0_values_2d / omega0
omega0_labels_2d = [rf"${val/omega0:.2f}\,\omega_0$" for val in omega0_factors_2d]
widths_2D_omega0_v0 = np.zeros((len(omega0_values_2d), len(v0_vec)))

N = 2**8
for i, omega0_val in enumerate(omega0_values_2d):
    for j, v0_test in enumerate(tqdm(v0_vec, desc=f"ω0={omega0_val:.2e}", leave=False)):
        widths_2D_omega0_v0[i, j] = float(final_state_probability_density(
            N, L_int, sigmaE, v0_test, omega0_val,
            vg, recoil, gamma_dB_per_cm
        )[5])


# Use log red-blue coloring (SymLogNorm) as before
finite_vals = widths_2D_omega0_v0[np.isfinite(widths_2D_omega0_v0)]
Wmin = float(np.nanmin(finite_vals))
Wmax = float(np.nanmax(finite_vals))
linthresh = 0.08 * max(abs(Wmin-initial_width), abs(Wmax - initial_width))
nrm = SymLogNorm(linthresh=linthresh, vmin=Wmin, vmax=Wmax, base=10)
cmap = "RdBu_r"
plt.figure(figsize=(8, 5))
im = plt.imshow(
    widths_2D_omega0_v0,
    aspect="auto",
    origin="lower",
    extent=[v0_vec.min()/c, v0_vec.max()/c, 0, len(omega0_values_2d)-1],
    cmap=cmap,
    norm=nrm
)
# Set fewer yticks for omega0 labels (e.g., 6 evenly spaced)
num_yticks = 6
ytick_indices = np.linspace(0, len(omega0_values_2d)-1, num_yticks, dtype=int)
ytick_vals = [omega0_values_2d[i] for i in ytick_indices]
ytick_labels = [f"{val/omega0:.2f} $\omega_0$" for val in ytick_vals]
plt.yticks(
    ticks=ytick_indices,
    labels=ytick_labels
)
# Set xticks for v0/c, evenly spaced
num_xticks = 6
xtick_vals = np.linspace(v0_vec.min()/c, v0_vec.max()/c, num_xticks)
plt.xticks(ticks=xtick_vals, labels=[f"{x:.6f}" for x in xtick_vals])
plt.xlabel("Electron velocity $v_0$ ($c$)")
plt.ylabel("Central frequency $\omega_0$")
plt.title("Final width vs $v_0$ and $\omega_0$\n($L_{int} = %.3g$ m, $\sigma_E = %.3g$ eV)" % (L_int, sigmaE))
plt.axvline(vg/c, color="red", linestyle="--", label=r"$v_0 = v_g$")
plt.legend()
cbar = plt.colorbar(im, label="Final width (eV)")
plt.tight_layout()
plt.show()

# Save the 2D omega0-v0 width data to CSV
df_omega0_v0 = pd.DataFrame(
    widths_2D_omega0_v0,
    index=[f"{val:.6e}" for val in omega0_values_2d],
    columns=[f"{val:.6e}" for val in v0_vec]
)
df_omega0_v0.index.name = "omega0"
df_omega0_v0.columns.name = "v0"
df_omega0_v0.to_csv("widths_2D_omega0_v0_less_resolution.csv")
# Save the last 2D omega0-v0 width figure as SVG
plt.savefig("widths_2D_omega0_v0_less_resolution.svg", format="svg")

# %%
