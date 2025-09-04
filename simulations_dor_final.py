# %%--------------------------------|TO DO's:|--------------------------------%% #
# * fix relativistic function
# * find relativistic recoil corrections
# * create relativistic numerical validation.
# * create a general relativisitc + loss density caluclation function.
# * 30dB cm losses simulation
### Q & A:
# should q0 be an input to the density function?
# %%=================================|SETUP:|===================================%% #
# %% import 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
from scipy.stats import linregress
import os, csv
import pandas as pd
from pathlib import Path
from tabulate import tabulate
# %% constants :
from scipy.constants import c, m_e as m, hbar, e, epsilon_0 as eps0
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
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
def λ(E_eV):
    """Calculate the de Broglie wavelength of an electron with energy E."""
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
def q0_func(omega0,v0,E_func,k_func):
    E0 = E_func(v0)                                 # central electron energy (J)
    k0 = k_func(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    epsilon = hbar*omega0/E0
    zeta = gamma/(gamma+1)
    sigma = -1/(gamma + 1)**2
    return k0*(zeta*epsilon + sigma*epsilon**2)
def v_g_func(omega0,v0,E_func,k_func):
    E0 = E_func(v0)                                 # central electron energy (J)
    k0 = k_func(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    zeta = gamma/(gamma+1)
    return E0/(k0*hbar) *(1/zeta)
def recoil_func(omega0,v0,E_func,k_func):
    E0 = E_func(v0)                                 # central electron energy (J)
    k0 = k_func(E0)                                 # central electron wavenumber (rad/m)
    gamma  = np.sqrt(1/(1 - (v0/c)**2))
    sigma = -1/(gamma + 1)**2
    return  k0*hbar**2 *sigma/E0**2
def q0_CLASSICAL(omega0, v0, E_function, k_function):
    return omega0/v0 + 1/(2*k(E(v0))*(v0)**2)
def v_g_CLASSICAL(omega0, v0, E_function, k_function):
    return v0
def recoil_CLASSICAL(omega0, v0, E_function, k_function):
    return -1/((k(E(v0))*v0**2))
# Density matrices
def final_state_probability_density(N,
                                    initial_width_eV,
                                    L_int,
                                    v_g_function,
                                    recoil_function,
                                    v0,
                                    omega0,
                                    k_function,
                                    E_function,
                                    grid_factor
                                    ):
    # initial_width is sigma in eV
    sigmaE = (initial_width_eV * e) # J

    E0 = E_function(v0)
    k0 = k_function(E0); k0_m_hw = k_function(E0 - hbar * omega0)
    
    q0 = k0 - k0_m_hw           # phase matching
    v_g = v_g_function(omega0, v0, E_function, k_function)
    recoil = recoil_function(omega0, v0, E_function, k_function)

    
    δω = np.linspace(-grid_factor * sigmaE / hbar, grid_factor * sigmaE / hbar, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-grid_factor * sigmaE, grid_factor * sigmaE, N)       # J
    dE   = δE_f[1] - δE_f[0]

    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing='ij')

    rho_i_2d = (1/np.sqrt(2*np.pi*sigmaE**2)) * np.exp(-(δE_f_grid + hbar*δω_grid)**2/(2*sigmaE**2))
    i0 = np.argmin(np.abs(δω))
    K = np.zeros_like(δω)
    K[i0] = 1.0 / dω
    rho_i_1d = np.sum(rho_i_2d * K[:, None], axis=0) * dω   # equals rho_i_2d[i0, :]
    rho_i_1d /= (np.sum(rho_i_1d) * dE)
    
    Delta_PM = ( k_function(E0 + δE_f_grid + hbar*δω_grid)
               - k_function(E0 + δE_f_grid - hbar*omega0)
               - (q0 + (δω_grid / v_g) + 0.5 * recoil * δω_grid**2) )

    kernel = np.sinc(Delta_PM * L_int / (2*np.pi)) 

    # Electron marginal over ω (normalized over J)
    rho_f = np.sum((rho_i_2d * kernel**2), axis=0) * dω
    rho_f /= np.sum(rho_f * dE)

    # Photon marginal over δE_f (normalized over rad/s)
    rho_f_p = np.sum((rho_i_2d * kernel**2), axis=1) * dE
    rho_f_p /= np.sum(rho_f_p * dω)

    # convert to eV:
    
    δE_f_eV      = δE_f / e
    rho_f_per_eV = rho_f / e
    rho_i_per_eV = rho_i_1d / e
    
    dE_eV = δE_f_eV[1] - δE_f_eV[0]
    rho_i_per_eV /= (np.sum(rho_i_per_eV) * dE_eV)
    rho_f_per_eV /= (np.sum(rho_f_per_eV) * dE_eV)
    final_width_eV = compute_FWHM(δE_f, rho_f)/e
    
    return δE_f_eV, rho_f_per_eV, final_width_eV, rho_i_per_eV, δω, rho_f_p, 
def final_state_probability_density_lossy(
                                        N,
                                        k,
                                        initial_width,
                                        L_int,
                                        v_g,
                                        v0,
                                        omega0,
                                        gamma_db_per_cm
                                          ):
    
    # ---------- setup (unchanged) ----------
    sigmaE = initial_width * e  # J

    E0 = 0.5 * m * v0**2
    k0 = k(E0); k0_m_hw = k(E0 - hbar * omega0)
    q0 = k0 - k0_m_hw
    recoil = -1 / (k0 * v0**2)  # applied to (δω')^2 per your change

    # global grids for ω and δE_f (same as before)
    δω  = np.linspace(-4 * sigmaE / hbar,  4 * sigmaE / hbar, N)
    dω  = δω[1] - δω[0]

    δE_f = np.linspace(-4 * sigmaE, 4 * sigmaE, N)  # J
    dE   = δE_f[1] - δE_f[0]

    # ---------- losses & Lorentzian width ----------
    # dB/cm (power) -> amplitude Np/m, then Γ = v_g * α  [rad/s]
    alpha_np_per_cm = np.log(10.0)/20.0 * gamma_db_per_cm   # amplitude attenuation (Np/cm)
    alpha_np_per_m  = alpha_np_per_cm * 100.0               # Np/m
    Gamma = v_g * alpha_np_per_m                            # rad/s
    if Gamma <= 0:
        Gamma = 1e-24

    # Band half-width in ω set by your global grid
    W = 4.0 * sigmaE / hbar  # rad/s; δω spans [-W, +W]

    # ---------- outputs ----------
    rho_f   = np.zeros_like(δE_f)  # electron marginal over J
    rho_f_p = np.zeros_like(δω)    # photon marginal over rad/s

    # ---------- per-ω local ω' = ω + u grid (plain sums) ----------
    # window in u: ± min(4Γ, W)  (4Γ = 8 * HWHM)
    U = min(4.0 * Gamma, W)
    # choose step to resolve the Lorentzian: du = min(global dω, Γ/8)
    du_target = Gamma / 8.0
    du = min(dω, du_target) if du_target > 0 else dω
    
    if du <= 0:
        du = dω
    # number of points on each side (odd count to include u=0)
    M_side = max(1, int(np.ceil(U / du)))
    u = np.linspace(-M_side * du, M_side * du, 2 * M_side + 1)  # shape (Mu,)
    u_col = u[:, None]                         # (Mu,1) for broadcasting
    δE_col = δE_f[None, :]                     # (1,N)

    # ---------- main loop over ω ----------
    for iω, ω in enumerate(δω):
        ωp = ω + u                              # local ω' grid around current ω, shape (Mu,)

        # Initial joint density ρ_i(E_f+ħω, E_f+ħω): depends only on (ω, E)
        rho_i_slice = (1/np.sqrt(2*np.pi*sigmaE**2)) * np.exp(-(δE_col + hbar*ω)**2/(2*sigmaE**2))  # (1,N)->(Mu,N)

        # Phase mismatch with q(ω') ≈ q0 + ω'/v_g and recoil on (ω')^2
        Delta_PM = ( k(E0 + δE_col + hbar*ω)
                   - k(E0 + δE_col - hbar*omega0)
                   - (q0 + (ωp[:, None] / v_g) + 0.5 * recoil * (ωp[:, None]**2)) )  # (Mu,N)

        kernel = np.sinc(Delta_PM * L_int / (2*np.pi))                               # (Mu,N)

        # Lorentzian L_Γ(ω' - ω)
        lorentz = (1/np.pi) * (Gamma/2.0) / ( (u_col**2) + (Gamma/2.0)**2 )          # (Mu,1)->(Mu,N)

        integrand = rho_i_slice * (kernel**2) * lorentz                              # (Mu,N)

        # ---- integrate by plain sums (Riemann rule) ----
        # electron marginal: integrate over u (ω') and then over ω (outer loop)
        rho_f += np.sum(integrand, axis=0) * du * dω                                 # -> (N,)

        # photon marginal at this ω: integrate over E and u; no outer dω factor here
        rho_f_p[iω] = np.sum(integrand) * dE * du

    # ---------- normalize ----------
    rho_f   /= np.sum(rho_f   * dE)
    rho_f_p /= np.sum(rho_f_p * dω)

    # ---------- initial 1D (same as before) ----------
    rho_i_1d = (1/np.sqrt(2*np.pi*sigmaE**2)) * np.exp(-(δE_f)**2/(2*sigmaE**2))

    # ---------- return per-eV (same as before) ----------
    δE_f_eV      = δE_f / e
    rho_f_per_eV = rho_f * e
    rho_i_per_eV = rho_i_1d * e

    dE_eV = δE_f_eV[1] - δE_f_eV[0]
    rho_i_per_eV /= (np.sum(rho_i_per_eV) * dE_eV)
    rho_f_per_eV /= (np.sum(rho_f_per_eV) * dE_eV)

    final_width_eV = compute_FWHM(δE_f, rho_f) / e
    return δE_f_eV, rho_f_per_eV, final_width_eV, rho_i_per_eV, δω, rho_f_p

    gamma = 1 / np.sqrt(1 - (v / c)**2)
    return (gamma - 1) * m * c**2 / e  # in eV
# plot functions:
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
def dispersion_plot(omega0, v0,v_g_function,recoil_function, E_function, k_function):
    # frequency grid around ω0
    omega_vec = np.linspace(0.005, 15, 200) * omega0
    δω_vec    = omega_vec - omega0

    # phase-matching linearization point (non-zero!)
    E0 = E_function(v0)                                 # central electron energy (J)
    q0_PM  = k_function(E0) - k_function(E0 - hbar*omega0)
    v_g     = v_g_function(omega0, v0, E_function, k_function)
    recoil = recoil_function(omega0, v0, E_function, k_function)

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
# simulation function:
def simulation_test(N, v0, sigma_eV , Lambda, L_interaction, k_function , E_function, Grid_factor):
    omega0  = 2 * np.pi * c / Lambda               # central angular frequency (rad/s)
    # Run the simulation with the given parameters
    δE_f_eV, rho_f_e, final_width_FWHM, rho_i_e, δω, rho_f_p = final_state_probability_density(
        N,
        sigma_eV,
        L_interaction,
        v_g_func,
        recoil_func,
        v0,
        omega0,
        k_function,
        E_function,
        Grid_factor
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    initial_width_FWHM = compute_FWHM(δE_f_eV, rho_i_e)
    # Electron probability density
    axes[0].plot(δE_f_eV, rho_i_e, label=f"initial FWHM = {initial_width_FWHM:.4f} eV")
    axes[0].plot(δE_f_eV, rho_f_e, label=f"final FWHM = {final_width_FWHM:.4f} eV")
    axes[0].set_xlabel("δE_f (eV)")
    axes[0].set_ylabel("Probability density (1/eV)")
    axes[0].legend()
    axes[0].set_title("Electron Probability Density")

    # Photon probability density
    axes[1].plot(δω, rho_f_p, label="Photon probability density")
    axes[1].set_xlabel("δω (rad/s)")
    axes[1].set_ylabel("Photon probability density (arb. units)")
    axes[1].legend()
    axes[1].set_title("Photon Probability Density vs δω")

    # Dispersion plot
    omega_vec = np.linspace(0.005, 3, 200) * omega0
    δω_vec    = omega_vec - omega0
    E0 = E_function(v0)
    q0_PM  = k_function(E0) - k_function(E0 - hbar*omega0)
    v_g     = v_g_func(omega0, v0, E_function, k_function)
    recoil  = recoil_func(omega0, v0, E_function, k_function)
    k_diff_vec = Δk(0.0, E0, δω_vec, omega0, k_function)
    q_vec      = q(δω_vec, q0_PM, v_g, recoil)
    axes[2].plot(k_diff_vec, omega_vec/(2*np.pi), 'b.', label="Δk(δω)")
    axes[2].plot(q_vec,      omega_vec/(2*np.pi), 'r.', label="q(δω)")
    axes[2].scatter([q0_PM], [omega0/(2*np.pi)], color='k', marker='x', s=80, label="Phase-matching point")
    axes[2].set_xlabel("Wavenumber (1/m)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_title("Relativistic Dispersion: Δk vs q(ω)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    # Simulation summary table
    v0_c = v0 / c
    E0_eV = E(v0) / e
    photon_energy_eV = hbar * omega0 / e
    photon_wavelength_m = Lambda
    L_interaction_m = L_interaction
    initial_FWHM_eV = 2*np.sqrt(2*np.log(2))*sigma_eV
    initial_energy_width_ratio = initial_FWHM_eV / E0_eV

    # Calculate final electron mean energy and width
    mean_final_energy_eV = np.sum(δE_f_eV * rho_f_e) * (δE_f_eV[1] - δE_f_eV[0])+ E0_eV-photon_energy_eV
    final_width_eV = final_width_FWHM
    final_energy_width_ratio = final_width_eV / E0_eV
    mean_energy_difference_eV = mean_final_energy_eV - E0_eV

    # Print summary table
    print("="*70)
    print("Simulation Summary:")
    print("="*70)
    print(f"{'Parameter':<35} | {'Value':>30}")
    print("-"*70)
    print(f"{'Electron velocity (v0/c)':<35}  | {v0_c:>30.6f}")
    print(f"{'Electron energy E(v0) [eV]':<35}  | {E0_eV:>30.4f}")
    print(f"{'Initial width FWHM [eV]':<35}  | {initial_FWHM_eV:>30.4f}")
    print(f"{'Final electron width FWHM [eV]':<35}  | {final_width_eV:>30.4f}")
    print(f"{'Photon energy [eV]':<35}  | {photon_energy_eV:>30.4f}")
    print(f"{'Photon wavelength [m]':<35}  | {photon_wavelength_m:>30.3e}")
    print(f"{'Interaction length [m]':<35}  | {L_interaction_m:>30.4e}")
    print(f"{'Final mean electron energy [eV]':<35}  | {mean_final_energy_eV:>30.4f}")
    print(f"{'Initial energy-width ratio':<35}  | {initial_energy_width_ratio:>30.4e}")
    print(f"{'Final energy-width ratio':<35}  | {final_energy_width_ratio:>30.4e}")
    print(f"{'Mean electron energy difference [eV]':<35} | {mean_energy_difference_eV:>30.4f}")
    print("="*70)
# plot functions:
# %%============================|TEST SIMULATIONS:|=============================%% # 
# %% test simulation :
# setup:
N = 2**11
v0 = 0.1 * c
lambda0 = 500e-9; omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01  # m
Delta_E_initial = 0.1 * (hbar * omega0) / e  # eV
omega0 = 2 * np.pi * c / lambda0
E0 = E_rel(v0)  # central electron energy (J)
simulation_test(N, v0, Delta_E_initial/2*np.log(2), lambda0 , L_int, k_rel ,E_rel ,7)


# %%  Width vs. Interaction Length Scan
N = 2**11
v0 = 0.1 * c
lambda0 = 500e-9; omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
L_int = 0.01  # m
Delta_E_initial = 0.1 * (hbar * omega0) / e  # eV
E0 = E_rel(v0)  # central electron energy (J)
initial_width = Delta_E_initial
L_num = 21  # Number of interaction lengths to test
L_int_vec = np.linspace(0.00002, 1.01, L_num)*L_int  # m
widths_L = []
probability = []
for L_int_test in tqdm(L_int_vec, desc="Scanning L_int"):
    width = final_state_probability_density(
        N,
        initial_width,
        L_int_test,
        lambda *args: v_g_func(omega0, v0, E_rel, k_rel),
        lambda *args: recoil_func(omega0, v0, E_rel, k_rel),
        v0,
        omega0,
        k_rel,
        E_rel,
        10
    )[2]
    widths_L.append(width)  # Store final width in eV

plt.figure()
plt.plot(L_int_vec * 1000, np.array(widths_L), ".-")
plt.plot(
    L_int_vec * 1000,
    [initial_width] * len(L_int_vec),
    label=f"Initial width = {initial_width:.4f} eV",
)
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length (mm)")
plt.legend()
plt.grid(True)
plt.show()
# %% v_g scan:
N = 2**11
v_g_num = 21  # Number of group velocities to test
omega0 =  2 * np.pi * c / 500e-9
v0 = 0.1*c
L_int = 0.01 #m  # Fixed interaction length for this scan
v_g_vec = np.unique(
    np.concatenate(
        [
            np.linspace(0.99, 1.01, v_g_num) * v0,
            np.linspace(0.999, 1.001, int(v_g_num / 2)) * v0,
        ]
    )
)
initial_width_eV = (0.1 * hbar * omega0)/e
widths_vg = []
for v_g_test in tqdm(v_g_vec, desc="Scanning v_g"):
    width = final_state_probability_density(
        N,
        initial_width_eV, 
        L_int,
        lambda *args: v_g_test,
        lambda *args: recoil_func(omega0, v0, E_rel, k_rel),
        v0, 
        omega0,
        k_rel,
        E_rel, 
        10
    )[2]
    widths_vg.append(width)  # Store final width in eV

plt.figure()
plt.plot(v_g_vec / c, widths_vg, ".-")
plt.plot(
    v_g_vec / c,
    [initial_width_eV] * len(v_g_vec),
    label=f"Initial width = {initial_width_eV:.4f} eV",
)
plt.axvline(v0 / c, color="red", linestyle="--", label="vg = v0")  # Add vertical line
plt.ylabel("Final Width (eV)")
plt.xlabel("Photon Group Velocity (c)")
plt.legend()
plt.grid(True)
plt.show()
# %% Plotting 2D graphs
ACCUM_CSV = "widths_2D_v0_LX.csv.csv"
expected = ["L_int_m", "v_g_m_per_s", "width"]
try:
    df = pd.read_csv(ACCUM_CSV)
    if not set(expected).issubset(df.columns):
        raise KeyError("bad header")
except Exception:
    # fallback: treat file as no-header and coerce to numeric
    df = pd.read_csv(ACCUM_CSV, header=None, names=expected)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

# remove any stray repeated header rows inside the file
for col in expected:
    df = df[pd.to_numeric(df[col], errors="coerce").notna()]
df = df.astype({"L_int_m": float, "v_g_m_per_s": float, "width": float})

grid = df.pivot_table(index="L_int_m", columns="v_g_m_per_s",
                      values="width", aggfunc="mean").sort_index().sort_index(axis=1)
L_all  = grid.index.values
vg_all = grid.columns.values
W      = grid.values
w = 10*np.log10(W/initial_width)

plt.figure()
# Force symmetric colorbar around zero
max_abs_value = max(abs(w.min()), abs(w.max()))
nrm = matplotlib.colors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)
cont0 = plt.imshow(
    w,
    extent=[vg_all.min()/c, vg_all.max()/c, L_all.min()*1e3, L_all.max()*1e3],
    origin='lower', aspect='auto', cmap=cm.RdBu_r, norm=nrm
)

# Add colorbar
cbar = plt.colorbar(cont0)
cbar.set_label('Final Width / Initial Width logscale', rotation=270, labelpad=20)

plt.xlabel('Photon Group Velocity (c)')
plt.ylabel('Interaction Length(mm)')
plt.xticks(np.arange(vg_all.min()/c, vg_all.max()/c + 5e-5, 5e-5), rotation=45)
plt.title('Final Width vs Photon Group Velocity and Interaction Length, initial width = {:.4f} eV'.format(initial_width))
plt.tight_layout()
plt.show()

# Energy-width ratio:
# min_width = np.min(widths_2D)
# min_idx = np.unravel_index(np.argmin(widths_2D), widths_2D.shape)
# min_L_int = L_int_vec[min_idx[0]]
# min_v_g = v_g_vec[min_idx[1]]
# print(f"Minimum width: {min_width:.6f} eV at L_int = {min_L_int:.6f} m, v_g = {min_v_g/c:.6f} c")
# print(f"Energy-width ratio: {hbar*min_width / (E_eV(v0)):.2f} rad/eV")

# %% v0 scan
v0_num = 21
N = 2**11
omega0 = 2 * np.pi * c / 500e-9

L_int = 0.01  # m
v0 = 0.1 * c

v0_vec = np.unique(
    np.concatenate([
        np.linspace(0.99, 1.01, v0_num) * v0,          # ±1%
        np.linspace(0.999, 1.001, v0_num // 2) * v0,    # ±0.1%
        [v0],                                           # exact center
    ])
)

widths_v0 = []
initial_width_eV = (0.1 * hbar * omega0) / e

for v0_test in tqdm(v0_vec, desc="Scanning v0"):
    width = final_state_probability_density(
        N,
        initial_width_eV,      # sigma in eV as in your function signature
        L_int,
        lambda *args: v_g_func(omega0, v0, E_rel, k_rel),  # keep v_g fixed
        lambda *args: recoil_func(omega0, v0_test, E_rel, k_rel),
        v0_test,
        omega0,
        k_rel, 
        E_rel,
        4
    )[2]
    widths_v0.append(width)

idx = np.argsort(v0_vec)
v0_sorted = v0_vec[idx] / c
widths_sorted = np.array(widths_v0)[idx]

plt.figure()
plt.plot(v0_sorted, widths_sorted, ".-")
plt.axhline(initial_width_eV, color="tab:orange",
            label=f"Initial width = {initial_width_eV:.4f} eV")

plt.xlabel("Electron Velocity v0 (c)")
plt.ylabel("Final Width (eV)")
plt.title(f"Final Width vs. Electron Velocity with L_int={L_int*1000} mm, v_g=0.1c")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %% 2D simulation: widths vs v_0 and L_int:
N = 2**11
L_num = 11
v_0_num = 11
v_0_vec = np.unique(
    np.concatenate([
        np.linspace(0.99, 1.01, v0_num) * v0,          # ±1%
        np.linspace(0.999, 1.001, v0_num // 2) * v0,    # ±0.1%
        [v0],                                           # exact center
    ])
)
L_int_vec = np.linspace(0.00002, 1.01, L_num)*L_int  # m
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L.csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
# Outer loop: L_int progress bar
for i, L_int_test in enumerate(tqdm(L_int_vec, desc="Scanning L_int", position=0)):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int_test:.5f}", leave=False, position=1)):
        width = float(final_state_probability_density(
            N,
            initial_width,
            L_int_test,
            lambda *args: v_g_func(omega0, v0, E_rel, k_rel),  
            lambda *args: recoil_func(omega0, v_0_test, E_rel, k_rel),
            v_0_test,
            omega0,
            k_rel,
            E_rel,
            10
        )[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int_test), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)

# %% Plotting 2D graphs
plot_v0_L_map("widths_2D_v0_L.csv", initial_width, v_g_func(omega0, v0, E_rel, k_rel), E_rel, 0)

# %%===============================|LOSS EFFECTS|==============================%% #

# %% Lossy simulation : 
N = 2**10
v0     = 0.1 * c                                # electron carrier velocity
lambda0 = 500e-9                                # central wavelength (m)
omega0  = 2 * np.pi * c / lambda0               # central angular frequency
v_g_fixed     = 1*v0                 # photon group velocity (m/s)
L_int = 0.01  # interaction length (m)  
deltaE_i = 0.1 * hbar * omega0
initial_width =  deltaE_i / (e*2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
gamma_db_per_cm = 100 # dB/cm loss

# sanity check dw ~ 2*pi*vg/L_int
(2*np.pi*v_g_fixed/L_int)/(np.log(10.0)/20.0 * gamma_db_per_cm*100*v_g_fixed)

δE_f_eV, rho_f_e, final_width, rho_i_e, δω, rho_f_p = final_state_probability_density_lossy(
                                            N,
                                            k,
                                            initial_width,
                                            L_int, v_g_fixed,
                                            v0,
                                            omega0,
                                            gamma_db_per_cm
                                            )    

plt.figure()
plt.plot(δE_f_eV, rho_i_e, label=f"initial σ = {initial_width:.4f} eV")
plt.plot(δE_f_eV, rho_f_e, label=f"final FWHM = {final_width:.4f} eV")
plt.xlabel("δE_f (eV)"); plt.ylabel("Probability density (1/eV)")
plt.legend()        
plt.title(f"Final Width with Losses (γ={gamma_db_per_cm:.2f}dB/cm)")
plt.show()
plt.figure()
plt.plot(δω, rho_f_p, label="Final photon probability density")
plt.xlabel("δω (rad/s)")
plt.ylabel("Photon probability density (arb. units)")
plt.legend()
plt.title("Final Photon Probability Density vs δω")
plt.show()
# %% 2D simulation: widths vs v_0 and L_int: (Lossy)
N = 2**11
L_num = 41
v_0_num = 41
vg_fixed = 0.1 * c  # Fixed group velocity for this scan
L_int_vec = np.linspace(0.0001, 0.01, L_num)              # m
v_0_vec  = np.linspace(0.999, 1.001, v_0_num) * vg_fixed  # m/s
gamma  = 100 #dB/cm
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L_(Lossy)_100dB.csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
for i, L_int in enumerate(tqdm(L_int_vec, desc="Scanning L_int")):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int:.5f}", leave=False)):
        width = float(final_state_probability_density_lossy(
                                                            N,
                                                            k,
                                                            initial_width,
                                                            L_int,
                                                            vg_fixed,
                                                            v_0_test,
                                                            omega0,
                                                            gamma
                                                            )[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)
# %% Plotting 2D graphs (Lossy)

#plot_v0_L_map("widths_2D_v0_L_(Lossy)_100dB.csv", initial_width, vg_fixed,E,3)
plot_v0_Lmap_COMPARISSON(
                        "widths_2D_v0_L_(Lossy)_3dB.csv",
                        "widths_2D_v0_L_(Lossy)_10dB.csv",
                        "widths_2D_v0_L_(Lossy)_30dB.csv",
                        "widths_2D_v0_L_(Lossy)_100dB.csv",
                        initial_width,
                        titles = ["3dB","10dB","30dB","100dB"] )
# %%===================|RELATIVISTIC CORRECTIONS EFFECTS|======================%% #
# %% finding recoil corrections:
N = 2**11
v0 = 0.001 * c 
lambda0 = 500e-9 
omega0 = 2 * np.pi * c / lambda0
# classical comparisson:
print("q0_test / q0_CLASSICAL:", q0_func(omega0, v0, E, k) / q0_CLASSICAL(omega0, v0, E, k))
print("vg_test / vg_CLASSICAL:", v_g_func(omega0, v0, E, k) / v_g_CLASSICAL(omega0, v0, E, k))
print("recoil_test / recoil_CLASSICAL:", recoil_func(omega0, v0, E, k) / recoil_CLASSICAL(omega0, v0, E, k))
# exact comparisson:


# Coeff_rel = Δk_PM_approx(2**10,E(v0),omega0,0,k_rel,E_rel)
# print((hbar*omega0/E0)**2)
# v_g_func(omega0, v0, E, k)/c
# v_g_CLASSICAL(omega0, v0, E, k)/c


# %% test Relativistic simulation  :
N = 2**11
v0     = 0.2 * c                                # electron carrier velocity
lambda0 = 500e-9                                # central wavelength (m)
omega0  = 2 * np.pi * c / lambda0               # central angular frequency (rad/s)
L_int = 0.02  # interaction length (m)
deltaE_i = 0.1 * hbar * omega0         
initial_width =  deltaE_i / (e*2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation

simulation_test(N, v0, deltaE_i, lambda0, L_int, k_rel, E_rel, 1e5)
# %% # %% 2D simulation: widths vs v_0 and L_int (Relativistic):
L_num = 11
v_0_num = 11
L_int = 3
vg_fixed_rel = v_g_func(omega0, v0, E_rel, k_rel)
L_int_vec = np.linspace(0.5, 1.5, L_num)*L_int
# m
v_0_vec  = np.linspace(0.999, 1.0001, v_0_num) * vg_fixed_rel  # m/s
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L_(Relativistic).csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
for i, L_int in enumerate(tqdm(L_int_vec, desc="Scanning L_int", position=0)):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int:.5f}", leave=False, position=1)):
        width = float(final_state_probability_density(
            N,
            initial_width,
            L_int,
            q0_func,
            lambda *args: vg_fixed_rel,
            recoil_func,
            v_0_test,
            omega0,
            k_rel,
            E_rel,
            4
        )[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)
# %% Plotting 2D graphs (relativistic)
plot_v0_L_map("widths_2D_v0_L_(Relativistic).csv", initial_width, vg_fixed_rel, E_rel, 0)
#
# %% ===========================|FINAL SIMULATIONS|============================%% #
# %%*************************|SEM|*************************%% #
# %% SEM setup (Si3N4, NIR)
λ_Si3N4 = 1.0e-6          # m     (wavelength)
q0_Si3N4 = 1.19e7         # rad/m (β0 = n_eff*2π/λ with n_eff≈1.90)
vg_Si3N4 = 1.55e8         # m/s   (v_g = c/n_g with n_g≈1.93)
recoil_Si3N4 = 4.4e-26    # s^2/m (q″ ≡ d^2β/dω^2; measured β2≈0.044 ps^2/m)
deltaE_SEM = 1.5          # eV    ( can be modified: typical: CFEG ~0.3 eV, Schottky ~0.5–0.7 eV, LaB6 ~1 eV)
loss_SEM_dB = 2.5         # dB/cm (propagation loss)
# %% 2D simulation: widths vs v_0 and L_int:
N = 2**11
L_num = 21
v_0_num = 21
vg_fixed = 0.1 * c  # Fixed group velocity for this scan
L_int_vec = np.linspace(0.000001, 0.5, L_num)              # m
v_0_vec  = np.linspace(0.9999, 1.00001, v_0_num) * vg_fixed  # m/s
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L.csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
for i, L_int in enumerate(tqdm(L_int_vec, desc="Scanning L_int")):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int:.5f}", leave=False)):
        width = float(final_state_probability_density(N, initial_width, L_int, vg_fixed, v_0_test, omega0)[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)
# %% Plotting 2D graphs 
# %%*************************|TEM|*************************%% #
# %% TEM (Silicon , Green Light)
# Setup:
# Electron:
deltaE_monochrometer_TEM = 100e-3  # eV    (monochromated TEM energy spread)
E0_eV_TEM = 80e3                      # eV    (ELECTRON ENERGY 80 keV)
v0_TEM = v_rel(E0_eV_TEM)             
# Photon & Waveguide :
λ_TEM = λ(0.8)                   # m     (wavelength) should be 0.8 - 1.2 eV
L_int = 4                     # interaction length (m)
omega0_TEM = 2 * np.pi * c / λ_TEM               # central angular frequency (rad/s)
initial_width_TEM = deltaE_monochrometer_TEM
initial_Energy_Width_ratio = initial_width_TEM / E0_eV_TEM
# simulation:
N = 2**12
simulation_test(N, v0_TEM, deltaE_monochrometer_TEM, λ_TEM, L_int, k_rel, E_rel, 5)
# %% Dispersion graph:
# To allow passing 0.2*v_g_func as an input, you need to wrap it in a lambda so it matches the expected function signature.
# For example:
dispersion_plot(
    omega0_TEM,
    v0_TEM,
    lambda *args: 1 * v_g_func(omega0_TEM, v0_TEM, E_rel, k_rel),
    recoil_func,
    E_rel,
    k_rel
)
# %% vg scan:
N = 2**11
v_g_num = 21  # Number of group velocities to test
v_g_mean = v_g_func(omega0_TEM, v0_TEM, E_rel, k_rel)
L_int = 4 #m Fixed interaction length for this scan
v_g_vec = np.unique(
    np.concatenate(
        [
            np.linspace(0.9999, 1.0001, v_g_num) * v_g_mean,
            np.linspace(0.99999, 1.00001, int(v_g_num / 2)) * v_g_mean,
        ]
    )
)
# L_int_vec = np.unique(np.concatenate([np.linspace(0.0025, 0.04, L_num), np.linspace(0.0025 , 0.00625, 6)]))

initial_width_eV = deltaE_monochrometer_TEM
widths_vg = []
for v_g_test in tqdm(v_g_vec, desc="Scanning v_g"):
    width = final_state_probability_density(
        N, 
        initial_width_eV,
        L_int,
        lambda *args: v_g_test,
        lambda *args: recoil_func(omega0, v0_TEM, E_rel, k_rel),
        v0_TEM, 
        omega0_TEM,
        k_rel, 
        E_rel, 
        10
    )[2]
    widths_vg.append(width)  # Store final width in eV

plt.figure()
plt.plot(v_g_vec / c, widths_vg, ".-")
plt.plot(
    v_g_vec / c,
    [initial_width_eV] * len(v_g_vec),
    label=f"Initial width = {initial_width_eV:.4f} eV",
)
plt.axvline(v_g_mean / c, color="red", linestyle="--", label="vg = vg_mean")  # Add vertical line
plt.ylabel("Final Width (eV)")
plt.xlabel("Photon Group Velocity (c)")
plt.legend()
plt.grid(True)
plt.show()
# %% v0 scan (TEM):
v0_num = 21
N = 2**11
# Build a sorted vector around v0 and ensure center value is present
v0_vec = np.unique(
    np.concatenate([
        np.linspace(0.9999, 1.0001, v0_num) * v0_TEM,          # ±1%
        np.linspace(0.99999, 1.00001, v0_num // 2) * v0_TEM,    # ±0.1%
        [v0_TEM],                                           # exact center
    ])
)

widths_v0 = []


for v0_test in tqdm(v0_vec, desc="Scanning v0"):
    width = final_state_probability_density(
        N,
        deltaE_monochrometer_TEM,      # sigma in eV as in your function signature
        L_int,
        lambda *args: v_g_func(omega0_TEM, v0_TEM, E_rel, k_rel),  # keep v_g fixed
        lambda *args: recoil_func(omega0_TEM, v0_TEM, E_rel, k_rel),
        v0_test,
        omega0_TEM,
        k_rel,
        E_rel,
        10
    )[2]
    widths_v0.append(width)

idx = np.argsort(v0_vec)
v0_sorted = v0_vec[idx] / c
widths_sorted = np.array(widths_v0)[idx]

plt.figure()
plt.plot(v0_sorted, widths_sorted, ".-")
plt.axhline(deltaE_monochrometer_TEM, color="tab:orange",
            label=f"Initial width = {initial_width_eV:.4f} eV")
plt.axvline(v0_TEM / c, color="red", linestyle="--", label="vg = vg_mean")  # Add vertical line
plt.xlabel("Electron Velocity v0 (c)")
plt.title(f"Final Width vs. Electron Velocity with L_int={L_int*1000} mm, v_g=0.1c")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %% interaction length scan:
N = 2**11
L_num = 11  # Number of interaction lengths to test
L_int_vec = np.linspace(0.002, 1.5, L_num)*L_int  # m
widths_L = []
probability = []
for L_int_test in tqdm(L_int_vec, desc="Scanning_L_int"):
    width = final_state_probability_density(
        N,
        deltaE_monochrometer_TEM,
        L_int_test,
        lambda *args: v_g_func(omega0_TEM, v0_TEM, E_rel, k_rel),
        lambda *args: recoil_func(omega0_TEM, v0_TEM, E_rel, k_rel),
        v0_TEM,
        omega0_TEM,
        k_rel,
        E_rel,
        10
    )[2]
    widths_L.append(width)  # Store final width in eV

plt.figure()
plt.plot(L_int_vec * 1000, np.array(widths_L)/initial_width, ".")
plt.plot(
    L_int_vec * 1000,
    [initial_width] * len(L_int_vec),
    label=f"Initial width = {initial_width:.4f} eV",
)
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length (mm)")
plt.legend()
plt.grid(True)
plt.show()
# %% 2D scan Width(v0,L_int)
N = 2**12
L_num = 5
v_0_num = 5
v0_vec = np.unique(
    np.concatenate([
        np.linspace(0.999, 1.001, v0_num) * v0_TEM,          # ±1%
        np.linspace(0.99999, 1.00001, v0_num // 2) * v0_TEM,    # ±0.1%
        [v0_TEM],                                           # exact center
    ])
)
L_int_vec = np.linspace(0.2, 1.5, L_num)*L_int  # m
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L_(TEM-UV).csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
for i, L_int_test in enumerate(tqdm(L_int_vec, desc="Scanning L_int", position=0)):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int_test:.5f}", leave=False, position=1)):
        width = float(final_state_probability_density(
            N,
            deltaE_monochrometer_TEM ,
            L_int_test,
            lambda *args: v_g_func(omega0_TEM, v0_TEM, E_rel, k_rel),
            lambda *args: recoil_func(omega0_TEM, v0_TEM, E_rel, k_rel) ,
            v_0_test,
            omega0_TEM,
            k_rel,
            E_rel,
            5
        )[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int_test), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)



# %% Plotting 2D graphs (TEM-UV)
plot_v0_L_map("widths_2D_v0_L_(TEM-UV).csv", deltaE_monochrometer_TEM, v_g_func(omega0_TEM, v0_TEM, E_rel, k_rel), E_rel, 0)


# %%
