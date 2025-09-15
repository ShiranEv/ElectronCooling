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
# %% constants :
from scipy.constants import c, m_e as m, hbar, e, epsilon_0 as eps0
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from scipy.interpolate import interp1d
import csv
# %% functions :
# Definitions
def k(E):
    return np.sqrt(2 * m * E) / hbar
def k_rel(E):
    return np.sqrt(E**2 + 2 * E * (m * c**2)) / (hbar * c)
def E(v0):
    """Calculate the energy of an electron with velocity v0."""
    return 0.5 * m * v0**2  # in Joules
def E_eV(v0):
    """Calculate the energy of an electron with velocity v0."""
    return 0.5 * m * v0**2 / e  # in eV
def E_rel(v):
    """Calculate the relativistic energy of an electron with velocity v."""
    gamma = 1 / np.sqrt(1 - (v**2 / c**2))
    return (gamma - 1) * m * c**2
def v_rel(E_eV):
    """Calculate the relativistic velocity of an electron with energy E."""
    E = E_eV * e  # Convert eV to Joules
    gamma = np.sqrt(1 + 2 * E / (m * c**2))
    return c * np.sqrt(1 - 1/gamma**2)
def v_func(E):
    gamma = np.sqrt(1 + 2 * E / (m * c**2))
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
def final_state_probability_density(N, L_int,sigmaE_eV,v0,omega0,v_g,recoil):
    grid_factor = 4
    sigmaE = sigmaE_eV * e  # J
    δω = np.linspace(-grid_factor * sigmaE / hbar, grid_factor * sigmaE / hbar, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-grid_factor * sigmaE, grid_factor * sigmaE, N)  # J
    dE = δE_f[1] - δE_f[0]
   
    energy_span = max(abs(δE_f))
    
    # nyquist = nyquist_rate(v0, L_int, energy_span)
    # if dω > nyquist:
    #     print(f"Warning: δω = {dω:.3e} > Nyquist rate = {nyquist:.3e} (aliasing may occur)")

    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing="ij")
    rho_i_2d = np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * sigmaE**2)) / np.sqrt(2 * np.pi * sigmaE**2)
    i0 = np.argmin(np.abs(δω))
    K = np.zeros_like(δω)
    K[i0] = 1.0 / dω
    rho_e_initial = np.sum(rho_i_2d * K[:, None], axis=0) * dω  # equals rho_i_2d[i0, :]
    rho_e_initial /= np.sum(rho_e_initial) * dE

    E0 = E_rel(v0)
    k0 = k_rel(E0)
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
    rho_f_total = factor * U_factor * (rho_i_2d * kernel**2)
    # rho_f_total =  (rho_i_2d * kernel**2)

    # Electron marginal over ω (normalized over J)
    rho_e = np.sum(rho_f_total, axis=0) * dω
    final_width_eV = compute_FWHM(δE_f, rho_e) / e

    p1 = np.sum(rho_e * dE)
    rho_e = rho_e / p1  # Normalize

    # Convert δE_f to eV for output
    δE_f_eV = δE_f / e
    rho_e_initial_eV = rho_e_initial * e  # Convert density to per eV
    rho_e_eV = rho_e * e                  # Convert density to per eV

    # Photon marginal over δE_f (normalized over rad/s)
    rho_p = np.sum(rho_f_total, axis=1) * dE
    rho_p /= np.sum(rho_p * dω)

    return rho_e_eV, rho_e_initial_eV, rho_p, final_width_eV, p1

def final_state_probability_density_loss(N, L_int, sigmaE_eV, v0, omega0, gamma_db_per_cm):
    grid_factor = 4
    sigmaE = sigmaE_eV * e  # J
    
    δω = np.linspace(-grid_factor * sigmaE / hbar, grid_factor * sigmaE / hbar, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-grid_factor * sigmaE, grid_factor * sigmaE, N)  # J
    dE = δE_f[1] - δE_f[0]
    #### is this correct?
    energy_span = max(abs(δE_f))
    ####
    # nyquist = nyquist_rate(v0, L_int, energy_span)
    # if dω > nyquist:
    #     print(f"Warning: δω = {dω:.3e} > Nyquist rate = {nyquist:.3e} (aliasing may occur)")

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
    vg = v_g_func(omega0, v0)
    recoil = recoil_func(omega0, v0)
    
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
            k_rel(E0 + δE_col + hbar * ω)
            - k_rel(E0 + δE_col - hbar * omega0)
            - (q0 + (ωp[:, None] / vg) + 0.5 * recoil * (ωp[:, None] ** 2))
        )

        kernel = (hbar * k_rel(E0 + δE_col + hbar * ω) / m) * np.sinc(Delta_PM * L_int / (2 * np.pi))

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
        #rho_f_p[iω] = np.sum(integrand * dE * du)
    rho_f_p = np.sum(integrand, axis=1) * dE * du
    # Compute p1 before any normalization
    p1 = np.sum(rho_f * dE)
    print(f"p1 = {p1}")
    rho_f /= np.sum(rho_f * dE) if np.sum(rho_f * dE) != 0 else 1.0
    rho_f_e = rho_f
    final_width_eV = compute_FWHM(δE_f, rho_f_e) / e
    # Initial 1D distribution
    rho_i_1d = (1 / np.sqrt(2 * np.pi * sigmaE**2)) * np.exp(-((δE_f) ** 2) / (2 * sigmaE**2))
    rho_i_1d = rho_i_1d * e  # Convert density to per eV
    # Apply the weighted combination (no normalization of rho_f)

    
    rho_f_e = compute_FWHM(δE_f, rho_f_e) / e
 
    # Normalize photon marginal using Riemann sum
    rho_f_p_sum = np.sum(rho_f_p * dω)
    rho_f_p /= rho_f_p_sum if rho_f_p_sum != 0 else 1.0
    # Convert outputs to eV units
    δE_f_eV = δE_f / e
    rho_i_e = rho_i_1d

    return δE_f_eV, δω, rho_i_e, rho_f_p, final_width_eV, p1
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
# %%************************************************************SEM setup************************************************************%% # 
# %% SEM setup 
v0 = 0.1 * c  # electron velocity
E0 = E_rel(v0)
lambda0 = 500e-9
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01  # m
sigmaE = 0.1 * hbar * omega0 / e
initial_width = sigmaE * 2 * np.sqrt(2 * np.log(2)) 

rho_e, rho_e_initial, rho_p, final_width_eV, p1 = final_state_probability_density(2**10, L_int, sigmaE, v0, omega0,
                                                                   v_g_func(omega0, v0), recoil_func(omega0, v0))
# Plot electron final and initial distributions

print(p1)
# δE_f grid in eV
N = len(rho_e)
grid_factor = 4
sigmaE_J = sigmaE * e  # Convert to J if needed
δE_f = np.linspace(-grid_factor * sigmaE_J, grid_factor * sigmaE_J, N)  # J
δE_f_eV = δE_f / e
δω = np.linspace(-grid_factor * sigmaE_J / hbar, grid_factor * sigmaE_J / hbar, N)


plt.figure(figsize=(8, 5))
plt.plot(δE_f_eV, rho_e, label="Final electron distribution ($\\rho_f$)")
plt.plot(δE_f_eV, rho_e_initial, label="Initial electron distribution ($\\rho_i$)", linestyle="--")
plt.xlabel("Energy deviation $\\delta E_f$ (eV)")
plt.ylabel("Probability density")
plt.title("Electron Energy Distributions")
plt.legend()
plt.tight_layout()
plt.show()

# Plot photon distribution
plt.figure(figsize=(8, 5))
# δω grid in rad/s

plt.plot(δω, rho_p, label="Photon distribution")
plt.xlabel("Frequency deviation $\\delta\\omega$ (rad/s)")
plt.ylabel("Probability density")
plt.title("Photon Frequency Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# %% L vs width Lossless:
L_num = 21  # Number of interaction lengths to test
N = 2**9
L_int_vec = np.linspace(0.0001, 0.01, L_num)  # m
widths_L = []
probability = []
for L_int_test in tqdm(L_int_vec, desc="Scanning L_int", position=0):
    width = final_state_probability_density(N, L_int_test, sigmaE, v0, omega0, v_g_func(omega0, v0), recoil_func(omega0, v0))[3]
    widths_L.append(width)  # Store final width in eV


# Find intersection of widths_L with initial_width
widths_L = np.array(widths_L)
L_int_vec = np.array(L_int_vec)
diff = widths_L - initial_width

# Find sign changes (crossings)
sign_changes = np.where(np.diff(np.sign(diff)))[0]
if len(sign_changes) > 0:
    # Linear interpolation for more accurate intersection
    idx = sign_changes[0]
    f_interp = interp1d(widths_L[idx:idx+2], L_int_vec[idx:idx+2])
    L_cross = f_interp(initial_width)
    print(f"Intersection at L = {L_cross:.6g} m (width = initial width)")
else:
    print("No intersection found between widths_L and initial_width.")
# Plot widths vs interaction length after the loop
plt.figure(figsize=(8, 5))
plt.plot(L_int_vec, widths_L, marker='o', label="Final width (eV)")
plt.axhline(initial_width, color='r', linestyle='--', label="Initial width (eV)")
if len(sign_changes) > 0:
    plt.axvline(L_cross, color='g', linestyle='--', label=f"L0 = {L_cross:.4g} m")
plt.xlabel("Interaction length $L_{int}$ (m)")
plt.ylabel("Width (eV)")
plt.title("Final Width vs Interaction Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %% L0 comparison
L0_theory = 4*E0*v0/(omega0*sigmaE*e)
L0  = 0.00051  # m
print(f"Theoretical L0 = {L0_theory:.6g} m, Simulated L0 = {L0:.6g} m, Ratio = {L0/L0_theory:.3f} ")
# %% L vs width with loss:
L_num = 11  # Number of interaction lengths to test
N = 2**10

L_int_vec = np.logspace(np.log10(0.0001), np.log10(0.02), L_num)  # m
widths_L = [[] for _ in range(3)]
for i, gamma_db_per_cm in enumerate([0, 30, 100]):
    for L_int_test in tqdm(L_int_vec, desc="Scanning L_int", position=0):
        width = final_state_probability_density_loss(N, L_int_test, sigmaE, v0, omega0, gamma_db_per_cm)[3]
        widths_L[i].append(width)  # Store final width in eV
plt.figure()
# %% plots

widths_L_total = [np.array(w) for w in widths_L]

# Calculate L0 for the vertical line (using the same formula as above)
L0_theory = 1.18 * 4 * E0 * v0 / (omega0 * sigmaE * e)

# Plot vertical line for L0


plt.figure(figsize=(8, 5))
colors = ['b', 'm', 'c']  # Distinct colors for the three loss cases
labels = [f"Final Width {gamma_db_per_cm} dB/cm" for gamma_db_per_cm in [0, 30, 100]]
for i, (color, label) in enumerate(zip(colors, labels)):
    plt.loglog(L_int_vec, widths_L_total[i], ".-", color=color, label=label)
plt.axhline(initial_width, color='r', linestyle='--', label="Initial width (eV)")
plt.axvline(L0, color='g', linestyle='--', label=f"L0  = {L0:.4g} m")
plt.axvline(L0_theory, color='y', linestyle='--', label=f"L0 (theory) = {L0_theory:.4g} m")
plt.loglog(L_int_vec, 1 / L_int_vec / 5000, ".-", color='orange', label=r"$\propto 1/L_{int}$")
plt.loglog(L_int_vec, 1 / np.sqrt(L_int_vec) / 100, ".-", color='k', label=r"$\propto 1/\sqrt{L_{int}}$")
plt.xlabel("Interaction Length $L_{int}$ (m)", fontsize=12)
plt.ylabel("Final Electron Energy Width (eV)", fontsize=12)
plt.title("Final Width vs Interaction Length (log-log)", fontsize=14)
plt.grid(True, which="both", ls="--")
plt.legend(title="Legend", fontsize=10)
plt.tight_layout()
plt.show()
      
# %%************************************************************ TEM setup************************************************************%% # 
# %% TEM setup
# Electron:
sigmaE = 100e-3  # eV    (monochromated TEM energy spread)
E0 = 80e3                      # eV    (ELECTRON ENERGY 80 keV)
v0 = v_rel(E0)             
# Photon & Waveguide :
λ_TEM = λ(0.8)                   # m     (wavelength) should be 0.8 - 1.2 eV
L_int = 3                   # interaction length (m)
omega0  = 2 * np.pi * c / λ_TEM               # central angular frequency (rad/s)
initial_width = sigmaE * 2 * np.sqrt(2 * np.log(2)) 

rho_e, rho_e_initial, rho_p, final_width_eV, p1 = final_state_probability_density(2**10, L_int, sigmaE, v0, omega0,
                                                                   v_g_func(omega0, v0), recoil_func(omega0, v0))
# Plot electron final and initial distributions

print(p1)
# δE_f grid in eV
N = len(rho_e)
grid_factor = 4
sigmaE_J = sigmaE * e  # Convert to J if needed
δE_f = np.linspace(-grid_factor * sigmaE_J, grid_factor * sigmaE_J, N)  # J
δE_f_eV = δE_f / e
δω = np.linspace(-grid_factor * sigmaE_J / hbar, grid_factor * sigmaE_J / hbar, N)


plt.figure(figsize=(8, 5))
plt.plot(δE_f_eV, rho_e, label="Final electron distribution ($\\rho_f$)")
plt.plot(δE_f_eV, rho_e_initial, label="Initial electron distribution ($\\rho_i$)", linestyle="--")
plt.xlabel("Energy deviation $\\delta E_f$ (eV)")
plt.ylabel("Probability density")
plt.title("Electron Energy Distributions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot photon distribution
plt.figure(figsize=(8, 5))
# δω grid in rad/s

plt.plot(δω, rho_p, label="Photon distribution")
plt.xlabel("Frequency deviation $\\delta\\omega$ (rad/s)")
plt.ylabel("Probability density")
plt.title("Photon Frequency Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %% TEM with losses:
    # Lossy TEM simulation: set loss value (e.g., 30 dB/cm)
gamma_db_per_cm = 30

rho_e, δω, rho_e_initial, rho_p, final_width_eV, p1 = final_state_probability_density_loss(2**10, L_int, sigmaE, v0, omega0,
                                                                   gamma_db_per_cm)

    # δE_f grid in eV (reuse N, grid_factor, sigmaE_J from above)
plt.figure(figsize=(8, 5))
plt.plot(δE_f_eV, rho_e, label="Final electron distribution (lossy, $\\rho_f$)")
plt.plot(δE_f_eV, rho_e_initial, label="Initial electron distribution ($\\rho_i$)", linestyle="--")
plt.xlabel("Energy deviation $\\delta E_f$ (eV)")
plt.ylabel("Probability density")
plt.title(f"Electron Energy Distributions (Loss = {gamma_db_per_cm} dB/cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(δω, rho_p, label="Photon distribution (lossy)")
plt.xlabel("Frequency deviation $\\delta\\omega$ (rad/s)")
plt.ylabel("Probability density")
plt.title(f"Photon Frequency Distribution (Loss = {gamma_db_per_cm} dB/cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%  L vs width :
L_num = 21  # Number of interaction lengths to test
N = 2**9
L_int_vec = np.linspace(0.1, 20, L_num)  # m
loss_values = [0, 30, 100]
widths_L = [[] for _ in loss_values]
results = []

for i, gamma_db_per_cm in enumerate(loss_values):
    for L_int_test in tqdm(L_int_vec, desc=f"Scanning L_int (Loss={gamma_db_per_cm} dB/cm)", position=0):
        width = final_state_probability_density_loss(N, L_int_test, sigmaE, v0, omega0, gamma_db_per_cm)[3]
        widths_L[i].append(width)  # Store final width in eV
        results.append({"L_int_m": L_int_test, "gamma_db_per_cm": gamma_db_per_cm, "width": width})

output_file = "tem_energy_width_vs_length_vs_loss.csv"
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L_int_m", "gamma_db_per_cm", "width"])
        writer.writeheader()
        writer.writerows(results)
else:
    # Optionally append new data (skip if not needed)
    with open(output_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L_int_m", "gamma_db_per_cm", "width"])
        writer.writerows(results)
        
        # Read widths vs length vs loss from CSV and plot all losses on the same figure
#%%
# Find intersection of widths_L with initial_width

        csv_file = "tem_energy_width_vs_length_vs_loss.csv"
        df = pd.read_csv(csv_file)

        loss_values = [0, 30, 100]
        plt.figure(figsize=(8, 5))
        colors = ['b', 'm', 'c']
        labels = [f"Final Width {gamma_db_per_cm} dB/cm" for gamma_db_per_cm in loss_values]

        for color, label, gamma_db_per_cm in zip(colors, labels, loss_values):
            group = df[df["gamma_db_per_cm"] == gamma_db_per_cm].sort_values("L_int_m")
            plt.plot(group["L_int_m"], group["width"], marker='o', label=label, color=color)

        plt.axhline(initial_width, color='r', linestyle='--', label="Initial width (eV)")
        plt.xlabel("Interaction length $L_{int}$ (m)")
        plt.ylabel("Width (eV)")
        plt.title("Final Width vs Interaction Length for Different Losses")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# %% L vs width log:
L_num = 11  # Number of interaction lengths to test
N = 2**9

L_int_vec = np.logspace(np.log10(0.1), np.log10(15), L_num)  # m
widths_L = [[] for _ in range(3)]
loss_values = [0, 30, 100]

# Prepare data for saving
results = []

for i, gamma_db_per_cm in enumerate(loss_values):
    for L_int_test in tqdm(L_int_vec, desc=f"Scanning L_int (Loss={gamma_db_per_cm} dB/cm)", position=0):
        width = final_state_probability_density_loss(N, L_int_test, sigmaE, v0, omega0, gamma_db_per_cm)[3]
        widths_L[i].append(width)  # Store final width in eV
        results.append({
            "L_int_m": L_int_test,
            "gamma_db_per_cm": gamma_db_per_cm,
            "width": width
        })

# Save results to CSV with a descriptive name
output_file = "tem_energy_width_vs_length_vs_loss.csv"
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L_int_m", "gamma_db_per_cm", "width"])
        writer.writeheader()
        writer.writerows(results)
else:
    # Append only new data (optional: skip if not needed)
    with open(output_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L_int_m", "gamma_db_per_cm", "width"])
        writer.writerows(results)

plt.figure()
# %% plots

# Load widths_L from CSV file
df = pd.read_csv("tem_energy_width_vs_length_vs_loss.csv")
widths_L_total = []
loss_values = [0, 30, 100]
for gamma_db_per_cm in loss_values:
    mask = df["gamma_db_per_cm"] == gamma_db_per_cm
    # Ensure sorting by L_int_m for correct plotting
    group = df[mask].sort_values("L_int_m")
    widths_L_total.append(group["width"].to_numpy())

# Calculate L0 for the vertical line (using the same formula as above)
L0_theory = 1.18 * 4 * E0 * v0 / (omega0 * sigmaE * e)

# Plot vertical line for L0


plt.figure(figsize=(8, 5))
colors = ['b', 'm', 'c']  # Distinct colors for the three loss cases
labels = [f"Final Width {gamma_db_per_cm} dB/cm" for gamma_db_per_cm in [0, 30, 100]]
for i, (color, label) in enumerate(zip(colors, labels)):
    plt.loglog(L_int_vec, widths_L_total[i], ".-", color=color, label=label)
plt.axhline(initial_width, color='r', linestyle='--', label="Initial width (eV)")
plt.axvline(L0, color='g', linestyle='--', label=f"L0  = {L0:.4g} m")
plt.axvline(L0_theory, color='y', linestyle='--', label=f"L0 (theory) = {L0_theory:.4g} m")
plt.loglog(L_int_vec, 1 / L_int_vec / 5000, ".-", color='orange', label=r"$\propto 1/L_{int}$")
plt.loglog(L_int_vec, 1 / np.sqrt(L_int_vec) / 100, ".-", color='k', label=r"$\propto 1/\sqrt{L_{int}}$")
plt.xlabel("Interaction Length $L_{int}$ (m)", fontsize=12)
plt.ylabel("Final Electron Energy Width (eV)", fontsize=12)
plt.title("Final Width vs Interaction Length (log-log)", fontsize=14)
plt.grid(True, which="both", ls="--")
plt.legend(title="Legend", fontsize=10)
plt.tight_layout()
plt.show()

# %%
