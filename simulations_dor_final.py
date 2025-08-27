# %%********************************|TO DO's:|**********************************%% #
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
def Δk(δE_f,E0,δω,omega0,k):
    k_E = k(δE_f + E0 - hbar*δω)
    k_E0 = k(E0 - hbar*omega0)
    return k_E - k_E0
def Δk_PM_approx(
                E0,
                ω0,
                delta_E,
                k_func, E_func):
    """
    Compute Taylor approximation coefficients of f(δE_f, δω) around (0,0) up to 2nd order.
    
    f(δE_f, δω) = k(δE_f + E0 - hbar*δω) - k(δE_f + E0 - hbar*ω0)
    
    Parameters:
    -----------
    δE_f : array_like
        Energy offset grid (1D array)
    δω : array_like  
        Frequency offset grid (1D array)
    E0 : float
        Reference energy
    ω0 : float
        Reference frequency
    hbar : float
        Reduced Planck constant
    k_func : callable
        Function k(E) to evaluate
        
    Returns:
    --------
    dict : Dictionary containing Taylor coefficients
        
        c00: f(0,0)
        c10: ∂f/∂δE_f at (0,0)  
        c01: ∂f/∂δω at (0,0)
        c20: (1/2) * ∂²f/∂δE_f² at (0,0)
        c11: ∂²f/∂δE_f∂δω at (0,0) 
        c02: (1/2) * ∂²f/∂δω² at (0,0)
    
    """
    
    sigmaE = delta_E 
    E0 = E_func(v0)

    
    δω = np.linspace(-4 * sigmaE / hbar, 4 * sigmaE / hbar, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-4 * sigmaE, 4 * sigmaE, N)       # J
    dE   = δE_f[1] - δE_f[0]

    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing='ij')
    
    # Create meshgrids
    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing='ij')
    
    # Compute f(δE_f, δω) on the grid
    f_values = k_func(δE_f_grid + E0 - hbar * δω_grid) - k_func(E0 - hbar * ω0)
    
    # Find center indices (corresponding to δE_f=0, δω=0)
    center_i = len(δω) // 2  # ω index
    center_j = len(δE_f) // 2  # E_f index
    
    # Grid spacings
    dω = δω[1] - δω[0]
    dE = δE_f[1] - δE_f[0]
    
    # Compute Taylor coefficients using finite differences
    
    # 0th order: f(0,0)
    c00 = f_values[center_i, center_j]
    
    # 1st order derivatives
    # ∂f/∂δE_f at (0,0)
    c10 = (f_values[center_i, center_j+1] - f_values[center_i, center_j-1]) / (2 * dE)
    
    # ∂f/∂δω at (0,0)  
    c01 = (f_values[center_i+1, center_j] - f_values[center_i-1, center_j]) / (2 * dω)
    
    # 2nd order derivatives
    # (1/2) * ∂²f/∂δE_f² at (0,0)
    c20 = 0.5 * (f_values[center_i, center_j+1] - 2*f_values[center_i, center_j] + 
                 f_values[center_i, center_j-1]) / (dE**2)
    
    # ∂²f/∂δE_f∂δω at (0,0)
    c11 = (f_values[center_i+1, center_j+1] - f_values[center_i+1, center_j-1] - 
           f_values[center_i-1, center_j+1] + f_values[center_i-1, center_j-1]) / (4 * dω * dE)
    
    # (1/2) * ∂²f/∂δω² at (0,0)
    c02 = 0.5 * (f_values[center_i+1, center_j] - 2*f_values[center_i, center_j] + 
                 f_values[center_i-1, center_j]) / (dω**2)
    
    return {
        'c00': c00,  # constant term
        'c10': c10,  # coefficient of δE_f
        'c01': c01,  # coefficient of δω
        'c20': c20,  # coefficient of δE_f²
        'c11': c11,  # coefficient of δE_f*δω
        'c02': c02   # coefficient of δω²
    }
#  photon disperssion coefficients functions:
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
def q(δω,q0,vg,recoil):
    return q0 + (δω / vg) + 0.5 * recoil * δω**2
def vg_OLD(omega0, v0, E_function, k_function):
    return v0
def recoil_OLD(omega0, v0, E_function, k_function):
    return -1/((k(E(v0))*v0**2))
# Density matrices
def final_state_probability_density(N,
                                    initial_width,
                                    L_int,
                                    q0,
                                    v_g_function,
                                    recoil_function,
                                    v0,
                                    omega0,
                                    k_function,
                                    E_function,
                                    grid_factor
                                    ):
    # initial_width is sigma in eV
    sigmaE = initial_width * e  # J

    E0 = E_function(v0)
    k0 = k_function(E0); k0_m_hw = k_function(E0 - hbar * omega0)
    
    q0 = k0 - k0_m_hw
    vg = v_g_function(omega0, v0, E_function, k_function)
    recoil = recoil_function(omega0, v0, E_function, k_function)

    
    δω = np.linspace(-grid_factor * sigmaE / hbar, grid_factor * sigmaE / hbar, N)
    dω = δω[1] - δω[0]
    δE_f = np.linspace(-grid_factor * sigmaE, grid_factor * sigmaE, N)       # J
    dE   = δE_f[1] - δE_f[0]

    δω_grid, δE_f_grid = np.meshgrid(δω, δE_f, indexing='ij')

    rho_i_2d = (1/np.sqrt(2*np.pi*sigmaE**2)) * np.exp(-(δE_f_grid + hbar*δω_grid)**2/(2*sigmaE**2))

    Delta_PM = ( k(E0 + δE_f_grid + hbar*δω_grid)
               - k(E0 + δE_f_grid - hbar*omega0)
               - (q0 + (δω_grid / vg) + 0.5 * recoil * δω_grid**2) )

    kernel = np.sinc(Delta_PM * L_int / (2*np.pi))

    # Electron marginal over ω (normalized over J)
    rho_f = np.sum((rho_i_2d * kernel**2), axis=0) * dω
    rho_f /= np.sum(rho_f * dE)

    # Photon marginal over δE_f (normalized over rad/s)
    rho_f_p = np.sum((rho_i_2d * kernel**2), axis=1) * dE
    rho_f_p /= np.sum(rho_f_p * dω)

    # Initial 1D (in J)
    rho_i_1d = (1/np.sqrt(2*np.pi*sigmaE**2)) * np.exp(-(δE_f)**2/(2*sigmaE**2))

    # ---- minimal change: return per-eV versions for plotting ----
    δE_f_eV      = δE_f / e
    rho_f_per_eV = rho_f * e
    rho_i_per_eV = rho_i_1d * e
    
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
# simulation function:
def simulation_test(N, v0, Delta_E_initial, Lambda, L_interaction,k , E, Grid_factor):
    
    omega0  = 2 * np.pi * c / Lambda               # central angular frequency (rad/s)
    initial_width =  Delta_E_initial / (e*2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    q0_OLD = omega0/v0 + 1/(2*k(E(v0))*(v0)**2)
     # Run the simulation with the given parameters
    δE_f_eV, rho_f_e, final_width, rho_i_e, δω, rho_f_p = final_state_probability_density(
                                                                                            N,
                                                                                            initial_width,
                                                                                            L_int,
                                                                                            q0_func,v_g_func ,
                                                                                            recoil_func,
                                                                                            v0, omega0,
                                                                                            k,E,4
                                                                                             )
    plt.figure()
    plt.plot(δE_f_eV, rho_i_e, label=f"initial σ = {initial_width:.4f} eV")
    plt.plot(δE_f_eV, rho_f_e, label=f"final FWHM = {final_width:.4f} eV")
    plt.xlabel("δE_f (eV)")
    plt.ylabel("Probability density (1/eV)")
    plt.legend()
    plt.show()

    # Plot photon probability density
    plt.figure()
    plt.plot(δω, rho_f_p, label="Photon probability density")
    plt.xlabel("δω (rad/s)")
    plt.ylabel("Photon probability density (arb. units)")
    plt.legend()
    plt.title("Photon Probability Density vs δω")
    plt.show()
    # Simulation summary table
    v0_c = v0 / c
    E0_eV = E(v0) / e
    photon_energy_eV = hbar * omega0 / e
    photon_wavelength_m = Lambda
    L_interaction_m = L_interaction
    initial_width_eV = initial_width
    initial_energy_width_ratio = initial_width_eV / E0_eV

    # Calculate final electron mean energy and width
    mean_final_energy_eV = np.sum(δE_f_eV * rho_f_e) * (δE_f_eV[1] - δE_f_eV[0])+ E0_eV-photon_energy_eV
    final_width_eV = final_width
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
    print(f"{'Initial width [eV]':<35}  | {initial_width_eV:>30.4f}")
    print(f"{'Initial energy-width ratio':<35}  | {initial_energy_width_ratio:>30.4e}")
    print(f"{'Photon energy [eV]':<35}  | {photon_energy_eV:>30.4f}")
    print(f"{'Photon wavelength [m]':<35}  | {photon_wavelength_m:>30.3e}")
    print(f"{'Interaction length [m]':<35}  | {L_interaction_m:>30.4e}")
    print(f"{'Final mean electron energy [eV]':<35}  | {mean_final_energy_eV:>30.4f}")
    print(f"{'Final electron width [eV]':<35}  | {final_width_eV:>30.4f}")
    print(f"{'Final energy-width ratio':<35}  | {final_energy_width_ratio:>30.4e}")
    print(f"{'Mean electron energy difference [eV]':<35} | {mean_energy_difference_eV:>30.4f}")
    print("="*70)
# plot functions:
def plot_v0_L_map(csv_path, initial_width, vg_fixed, E, Loss):
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

    # --- COLOR FIX ---
    # BLUE = narrower (W < 0),
    # RED = broader (W > 0).
    
    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)
    if zmin < 0 and zmax > 0:
        nrm = matplotlib.colors.TwoSlopeNorm(vmin=zmin, vcenter=0.0, vmax=zmax)
    else:
    # fallback: simple Normalize without a center at 0
        nrm = matplotlib.colors.Normalize(vmin=zmin, vmax=zmax)

    cmap = "RdBu_r"  # <-- Reversed to match requested sense

    

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
    E_J  = E(min_v0)  # in Joules
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
# %%============================|TEST SIMULATIONS:|=============================%% # 
# %% test simulation :
N = 2**11
v0 = 0.1 * c 
lambda0 = 500e-9 
L_int = 0.01  # m

simulation_test(N, v0, 0.1*(hbar*2 * np.pi * c /lambda0), lambda0 , L_int, k ,E ,5)
# %% coeffiecient validation :
# validation with classical case:
v0 = 0.1 * c  # electron carrier velocity
omega0 = 2 * np.pi * c / 500e-9  # central angular frequency (rad/s)

q0_classical = omega0 / v0 + (1/(2*k(E(v0))))*( omega0 / v0)**2
vg_classical = v0
recoil_classical = -1/(k(E(v0))*v0**2)

q0_analytic = q0_func(omega0,v0,E_rel,k_rel)
vg_analytic = v_g_func(omega0,v0,E_rel,k_rel)
recoil_analytic = recoil_func(omega0,v0,E_rel,k_rel)

print("--------------------------------------------------------")
print("classical regime:")
print("--------------------------------------------------------")
print("q0_rel / q0_classical:", q0_rel / q0_classical)
print("vg_rel / vg_classical:", vg_rel / vg_classical)
print("recoil_rel / recoil_classical:", recoil_rel / recoil_classical)
print("--------------------------------------------------------")
print("relativistic regime:")
print("--------------------------------------------------------")
# validation in relativistic regime:
q0_rel = k(E(v0)) - k(E(v0) - hbar * omega0)
print("q0_rel / q0_analytic:", q0_rel / q0_analytic)
# Plot Δk, q, and their difference on the same graph
δω_plot = np.linspace(0, 100 * hbar * omega0, 1000) / hbar  # rad/s
Δk_vals = Δk(0, E(v0), δω_plot, omega0, k)
q_vals = q(δω_plot, q0_analytic, vg_analytic, recoil_analytic)
diff_vals = Δk_vals - q_vals

plt.figure(figsize=(8, 5))
plt.plot(δω_plot, Δk_vals, label=r'$\Delta k$')
plt.plot(δω_plot, q_vals, label=r'$q$')
plt.plot(δω_plot, diff_vals, label=r'$\Delta k - q$')
plt.xlabel(r'$\delta\omega$ (rad/s)')
plt.ylabel('Value (1/m)')
plt.legend()
plt.title(r'Comparison of $\Delta k$, $q$, and $\Delta k - q$')
plt.grid(True)
plt.tight_layout()
plt.show()
# %% recoil scan?
# v0 = 0.1 * c  # electron carrier velocity
# E0 = 0.5 * m * v0**2  # central electron energy (J)

# lambda0 = 500e-9  # central wavelength (m)
# omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
# v_g = v0  # photon group velocity (m/s)

# deltaE = 0.1 * hbar * omega0  # energy spread (J)
# N = 2**12
# # --- ENERGY GRID
# N_E = N  # number of energy points

# E_min = E0 - 10 * deltaE
# E_max = E0 + 10 * deltaE

# E_f = np.linspace(E_min, E_max, N_E)
# dE = E_f[1] - E_f[0]
# energy_span = E_max - E_min

# δE_f = E_f - E0

# N_ω = N
# omega_span = 10 * deltaE / hbar  # Narrow span around ω₀
# ω_min = max(omega0 - omega_span / 2, 0 * omega0)  # Start from ω₀ - span/2
# ω_max = omega0 + omega_span / 2  # End at ω₀ + span/2
# ω_vec = np.linspace(ω_min, ω_max, N_ω)
# dω = ω_vec[1] - ω_vec[0]

# δω = ω_vec - omega0

# δω_grid, δE_f_grid = np.meshgrid(δω, δE_f)
# q0 = k_rel(E0) - k_rel(E0 - hbar * omega0)

# recoil_vec = np.linspace(-0.01, 0.01, 2)  # Recoil vector in m/s
# res = [
#     [
#         k_rel(E0 + hbar * δω_) - k_rel(E0 - hbar * omega0) - (q0 + (δω_ / v_g) + 0.5 * recoil * δω_**2)
#         for recoil in recoil_vec
#     ]
#     for δω_ in δω
# ]
# plt.figure()
# plt.imshow(
#     res,
#     extent=[δω.min(), δω.max(), recoil_vec.min() / e, recoil_vec.max() / e],
#     origin="lower",
#     aspect="auto",
#     cmap="viridis",
# )
# plt.xlabel("")
# # plt.xticks(np.arange(0.099, 0.102, 0.001), rotation=45)
# plt.ylabel("")
# plt.show()


# # %% v_g scan:
# v_g_num = 11  # Number of group velocities to test
# # Combine and sort unique values
# v_g_vec = np.unique(
#     np.concatenate(
#         [
#             np.linspace(0.099, 0.101, v_g_num) * c,
#             np.linspace(0.0999, 0.1001, int(v_g_num / 2)) * c,
#         ]
#     )
# )
# widths_vg = []

# for v_g_test in v_g_vec:
#     width = final_state_probability_density(
#         N, initial_width,
#         L_int, v_g_test,
#         v0, omega0
#     )[2]
#     widths_vg.append(width)  # Store final width in eV

# plt.figure()
# plt.plot(v_g_vec / c, widths_vg, ".-")
# plt.plot(
#     v_g_vec / c,
#     [initial_width] * len(v_g_vec),
#     label=f"Initial width = {initial_width:.4f} eV",
# )
# plt.xlabel("Photon Group Velocity (c)")
# plt.ylabel("Final Width (eV)")
# plt.title(f"Final Width vs. Photon Group Velocity with L_int={L_int*1000} mm")
# plt.xticks(np.arange(0.099, 0.102, 0.001), rotation=45)
# plt.legend()
# plt.show()

# %%  Width vs. Interaction Length Scan
# L_num = 11  # Number of interaction lengths to test
# L_int_vec = np.linspace(0.001, 0.01, L_num)  # m
# # L_int_vec = np.unique(np.concatenate([np.linspace(0.0025, 0.04, L_num), np.linspace(0.0025 , 0.00625, 6)]))
# v_g = 0.1 * c  # Fixed group velocity for this scan

# widths_L = []
# probability = []
# for L_int_test in L_int_vec:
#     width = final_state_probability_density(
#         N, initial_width, L_int_test, v_g, v0, omega0
#     )[2]
    
#     widths_L.append(width)  # Store final width in eV

# plt.figure()
# plt.plot(L_int_vec * 1000, widths_L, ".")
# plt.plot(
#     L_int_vec * 1000,
#     [initial_width] * len(L_int_vec),
#     label=f"Initial width = {initial_width:.4f} eV",
# )
# plt.ylabel("Final Width (eV)")
# plt.xlabel("Interaction Length (mm)")
# plt.legend()
# plt.show()

# %% 2D  widths vs v_g and L_int
L_num = 21
v_g_num = 21
L_int_vec = np.linspace(0.000001, 0.02, L_num)       # m
v_g_vec  = np.linspace(0.09992, 0.10002, v_g_num) * c  # m/s
widths_2D = np.zeros((len(L_int_vec), len(v_g_vec)))
ACCUM_CSV = "widths_2D_vg_L3.csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_g_m_per_s,width\n")
_rows = []
for i, L_int in enumerate(tqdm(L_int_vec, desc="Scanning L_int")):
    for j, v_g in enumerate(tqdm(v_g_vec, desc=f"Scanning v_g for L_int={L_int:.5f}", leave=False)):
        width = float(final_state_probability_density(N, initial_width, L_int, v_g, v0, omega0)[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int), float(v_g), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_g_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)


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
v0_num = 11  # Number of v0 values to test
# Scan v0 around 0.1c (same as v_g)
v0_vec = np.unique(
    np.concatenate(
        [
            np.linspace(0.099, 0.101, v0_num) * c,
            np.linspace(0.0999, 0.1001, int(v0_num / 2)) * c,
        ]
    )
)
widths_v0 = []

v_g_fixed = 0.1 * c  # Keep v_g fixed at 0.1c

for v0_test in v0_vec:
    width = final_state_probability_density(N,k, initial_width, L_int, v_g_fixed, v0_test, omega0)[2]
    widths_v0.append(width)  # Store final width in eV

plt.figure()
plt.plot(v0_vec / c, widths_v0, ".-")
plt.plot(
    v0_vec / c,
    [initial_width] * len(v0_vec),
    label=f"Initial width = {initial_width:.4f} eV",
)
plt.xlabel("Electron Velocity v0 (c)")
plt.ylabel("Final Width (eV)")
plt.title(f"Final Width vs. Electron Velocity with L_int={L_int*1000} mm, v_g=0.1c")
plt.xticks(np.arange(0.099, 0.1015, 0.0005), rotation=45)
plt.legend()
plt.show()



# SEM parameters:

# %% 2D simulation: widths vs v_0 and L_int:
N = 2**10
L_num = 31
v_0_num = 31
vg_fixed = 0.1 * c  # Fixed group velocity for this scan
L_int_vec = np.linspace(0.0001, 0.01, L_num)              # m
v_0_vec  = np.linspace(0.999, 1.001, v_0_num) * vg_fixed  # m/s
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
plot_v0_L_map("widths_2D_v0_L.csv", initial_width, vg_fixed,E)
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
v0 = 0.1 * c                                # electron carrier velocity
E0 = E(v0)                                 # central electron energy (J)
k0 = k(E0)                                 # central electron wavenumber (rad/m)
lambda0 = 500e-9                                # central wavelength (m)
omega0  = 2 * np.pi * c / lambda0               # central angular frequency (rad/s)
gamma_test  = np.sqrt(1/(1 - (v0/c)**2))
epsilon = hbar*omega0/E0
zeta = gamma_test/(gamma_test+1)
alpha  = (gamma_test-1)/(gamma_test+1)
sigma = -1/(gamma_test + 1)**2
q0_test = k0*(zeta*epsilon + sigma*epsilon**2)
vg_test = E0/(k0*hbar) *(1/zeta)
recoil_test = k0*hbar**2 *sigma/E0**2
# test:
q0_OLD = omega0/v0 + 1/(2*k0*(v0)**2)
vg_OLD = v0
recoil_old = -1/((k0*v0**2))
print("q0_test / q0_OLD:", q0_test / q0_OLD)
print("vg_test / vg_OLD:", vg_test / vg_OLD)
print("recoil_test / recoil_old:", recoil_test / recoil_old)


print((hbar*omega0/E0)**2)


# %% test Relativistic simulation  :
N = 2**13
v0     = 0.1 * c                                # electron carrier velocity
lambda0 = 500e-9                                # central wavelength (m)
omega0  = 2 * np.pi * c / lambda0               # central angular frequency (rad/s)
L_int = 0.01  # interaction length (m)
deltaE_i = 0.1 * hbar * omega0         
initial_width =  deltaE_i / (e*2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation

simulation_test(N, v0, deltaE_i, lambda0, L_int, k_rel, E_rel, 25)
# %% # %% 2D simulation: widths vs v_0 and L_int (Relativistic):
L_num = 11
v_0_num = 11
vg_fixed_rel = 0.1 * c  # Fixed group velocity for this scan
L_int_vec = np.linspace(0.1, 1, L_num)       # m
v_0_vec  = np.linspace(0.9, 1.0001, v_0_num) * vg_fixed_rel  # m/s
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L_(Relativistic).csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
for i, L_int in enumerate(tqdm(L_int_vec, desc="Scanning L_int")):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int:.5f}", leave=False)):
        width = float(final_state_probability_density(N, initial_width, L_int,  vg_fixed,v_0_test, omega0)[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)
# %% Plotting 2D graphs (relativistic)
plot_v0_L_map("widths_2D_v0_L_(Relativistic).csv", initial_width, vg_fixed,E)
#
# %% ===========================|FINAL SIMULATIONS|============================%% #
# %%--------------------|SEM|--------------------%% #
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
# %%--------------------|TEM|--------------------%% #
# %% TEM (Silicon , Green Light)
# Electron:
deltaE_monochrometer_TEM = 25e-3  # eV    (monochromated TEM energy spread)
E0_eV_TEM = 80e3                      # eV    (ELECTRON ENERGY 80 keV)
v0_TEM = v_rel(E0_eV_TEM)             
# Photon & Waveguide :
λ_TEM = λ(0.8)                   # m     (wavelength) should be 0.8 - 1.2 eV
L_int = 0.1                    # interaction length (m)
omega0_TEM  = 2 * np.pi * c / λ_TEM               # central angular frequency (rad/s)


beta_TEM = v0_TEM / c             # electron velocity (β = v/c)
gamma_TEM = np.sqrt(1 / (1 - beta_TEM**2))
initial_width =   deltaE_monochrometer_TEM / (e*2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
initial_Energy_Width_ratio = initial_width / E0_eV_TEM
N = 2**10
simulation_test(N, v0_TEM, deltaE_monochrometer_TEM*e, λ_TEM, L_int, k_rel, E_rel, 85)

# %% 2D plot of widths vs v_0 and L_int (TEM-UV)
# 
L_num = 21
v_0_num = 21
vg_fixed = 0.1 * c  # Fixed group velocity for this scan
L_int_vec = np.linspace(0.000001, 0.02, L_num)       # m
v_0_vec  = np.linspace(0.0999, 0.10008, v_0_num) * c  # m/s
widths_2D = np.zeros((len(L_int_vec), len(v_0_vec)))
ACCUM_CSV = "widths_2D_v0_L_(TEM-UV).csv"
file_exists = Path(ACCUM_CSV).exists()   # capture BEFORE writing
if not file_exists:
    with open(ACCUM_CSV, "w") as f:
        f.write("L_int_m,v_0_m_per_s,width\n")
_rows = []
for i, L_int in enumerate(tqdm(L_int_vec, desc="Scanning L_int")):
    for j, v_0_test in enumerate(tqdm(v_0_vec, desc=f"Scanning v_0 for L_int={L_int:.5f}", leave=False)):
        width = float(final_state_probability_density(k, initial_width, L_int,  vg_fixed,v_0_test, omega0)[2])
        widths_2D[i, j] = width
        _rows.append((float(L_int), float(v_0_test), width))

df = pd.DataFrame(_rows, columns=["L_int_m", "v_0_m_per_s", "width"])
df.to_csv(ACCUM_CSV, mode="a", index=False, header=not file_exists)
# %% Plotting 2D graphs (TEM-UV)
ACCUM_CSV = "widths_2D_v0_L_(TEM-UV).csv"
expected = ["L_int_m", "v_0_m_per_s", "width"]
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
df = df.astype({"L_int_m": float, "v_0_m_per_s": float, "width": float})

grid = df.pivot_table(index="L_int_m", columns="v_0_m_per_s",
                      values="width", aggfunc="mean").sort_index().sort_index(axis=1)
L_all  = grid.index.values
v0_all = grid.columns.values
W      = grid.values
w = 10*np.log10(W/initial_width)

plt.figure()
# Force symmetric colorbar around zero
max_abs_value = max(abs(w.min()), abs(w.max()))
nrm = matplotlib.colors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)
cont0 = plt.imshow(
    w,
    extent=[v0_all.min()/c, v0_all.max()/c, L_all.min()*1e3, L_all.max()*1e3],
    origin='lower', aspect='auto', cmap=cm.RdBu_r, norm=nrm
)

# Add colorbar
cbar = plt.colorbar(cont0)
cbar.set_label('Final Width / Initial Width logscale', rotation=270, labelpad=20)
plt.xlabel('electron initial velocity (c)')
plt.ylabel('Interaction Length(mm)')
plt.xticks(np.arange(v0_all.min()/c, v0_all.max()/c + 5e-5, 5e-5), rotation=45)
plt.title('Final Width vs Electron Initial Velocity and Interaction Length, initial width = {:.4f} eV'.format(initial_width))
plt.tight_layout()
plt.show()

# Energy-width ratio:
min_width = np.min(widths_2D)
min_idx = np.unravel_index(np.argmin(widths_2D), widths_2D.shape)
min_L_int = L_int_vec[min_idx[0]]
min_v_0 = v_0_vec[min_idx[1]]
print(f"Minimum width: {min_width:.6f} eV at L_int = {min_L_int:.6f} m, v_0 = {min_v_0/c:.6f} c")
print(f"Energy-width ratio: {min_width / (E_eV(min_v_0)):.2e} rad/eV")


# %%


# %%
