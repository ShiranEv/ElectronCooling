#%% 
import numpy as np
import matplotlib.pyplot as plt
#%%
# from scipy.integrate import np.trapz
# Physical constants
c = 2.99792458e8  # m/s
m = 9.109383e-31  # kg
e = 1.60217662e-19  # C
hbar = 1.054571800e-34  # J⋅s
eps0 = 8.854187817e-12  # F/m 

#
def k(E):
    """Electron momentum from energy"""
    return np.sqrt(2 * m * E / hbar**2)

# System parameters
L = 2e-3  # interaction length (2mm) - more realistic

# Electron parameters
v0 = 0.25 * c  # electron carrier velocity # SAME VALUE AS IN 1st SIMULATION
E0 = 0.5 * m * v0**2  # central electron energy
deltaE = 0.1 * E0  # energy spread

# for later analysis
dbWaveLength = 2*np.pi * hbar*v0 /E0 # de Broglie wavelength
L_critical = dbWaveLength *(np.pi/4)* (E0/deltaE)**2  # critical length for wavefunction spread
#
k0 = k(E0)  # electron wavevector

# Photon parameters
lambda0 = 500e-9  # wavelength
omega0 = 2 * np.pi * c / lambda0  # central frequency
v_g = 0.25*c  # group velocity for free space photons
q0 = -(omega0/v0 + (1/(2*k0)) * (omega0/v0)**2)  # proper phase matching
recoil = -1/(k0*v0**2)  # second derivative (recoil term)

# Energy grid - narrower range around phase matching region
N_E = 1000
E_center = E0   # Expected center after photon emission
E_min, E_max = 0.1*E0,2*E0
Ef_vec = np.linspace(E_min, E_max, N_E)
dE = Ef_vec[1] - Ef_vec[0]

# Frequency grid - focused around resonance
N_omega = 1000  # Number of frequency points
omega_spread = 4*deltaE/hbar  # Match energy spread in frequency units
omega_min, omega_max = omega0 - omega_spread, omega0 + omega_spread
omega_vec = np.linspace(omega_min, omega_max, N_omega)
domega = omega_vec[1] - omega_vec[0]

def v(E):
    """Electron velocity from energy"""
    return hbar * k(E) / m

def q(omega):
    """Photon momentum with proper phase matching expansion"""
    delta_omega = omega - omega0
    return q0 + (1/v_g)*delta_omega + (1/2)*recoil*delta_omega**2 
    
# def psi_i(E):
#     """Initial electron wavefunction - Gaussian distribution"""
#     normalization = 1 / np.sqrt(np.sqrt(2 * np.pi) * deltaE)
#     return normalization * np.exp(-(E - E0)**2 / (2 * deltaE**2))
psi_i = (1 / (np.pi * deltaE**2)**0.25) * np.exp(-(Ef_vec - E0)**2 / (2 * deltaE**2))
psi_i /= np.linalg.norm(psi_i)  # normalize

def width_calculation(prob_density, energy_grid):
    """Calculate FWHM of probability distribution"""
    max_prob = np.max(prob_density)
    half_max_indices = np.where(prob_density >= max_prob / 2)[0]
    if len(half_max_indices) > 1:
        return energy_grid[half_max_indices[-1]] - energy_grid[half_max_indices[0]]
    else:
        return 0

# Calculate initial distribution
psi_i_squared = np.abs(psi_i)**2
initial_width = width_calculation(psi_i_squared, Ef_vec)
prob_Ei = psi_i*np.conj(psi_i)

initial_width = width_calculation(prob_Ei, Ef_vec)

# Convert energies to eV for display
Ef_vec_eV = Ef_vec / e  # Convert from Joules to eV
E0_eV = E0 / e
initial_width_eV = initial_width / e

print(f"Initial energy width: {initial_width_eV:.3f} eV ({initial_width/E0:.3f} E0)")
print(f"Central energy E0: {E0_eV:.1f} eV")

# # Plot with corrected units
# plt.figure(figsize=(8,6))
# plt.plot(Ef_vec_eV, psi_i*np.conj(psi_i), label='Initial Electron Wavefunction')
# plt.xlabel(r'$E_i$ (eV)')
# plt.ylabel(r'Probability density')
# plt.title('Initial Electron State, L = %.2f [mm], width = %.2f eV (%.3f E0)'  % (L*1e3, initial_width_eV, initial_width/E0))
# plt.grid(True)
# plt.legend()
# plt.show()

print("Computing final distribution...")
#%%
print("Computing final distribution...")

Phi = np.zeros((N_E, N_omega), dtype=complex)
factor = (e**2 *hbar*L**2)/(2*omega0*eps0)  # Normalize factor to avoid overflow

for i, Ef in enumerate(Ef_vec):
    for j, wq in enumerate(omega_vec):
        Ei = Ef + hbar*wq  # conservation of Energy
        if Ei <E_min or Ei > E_max:
            continue

        # Find nearest Ei index using proper interpolation
        Ei_index = np.argmin(np.abs(Ef_vec - Ei))
        if Ei_index >= N_E:
            continue
        
        phase_mismatch = (k(Ei) - k(Ef) -  (q(wq)/hbar)) 
        sinc_arg = (phase_mismatch * L) / 2
    
        sinc_val = np.sinc(sinc_arg / np.pi)
        
        # Use the probability density directly, not squared
        Phi[i, j] =  sinc_val**2 * psi_i_squared[Ei_index]

print(f"Phi range: [{np.min(np.abs(Phi)):.2e}, {np.max(np.abs(Phi)):.2e}]")


prob_Ef = np.zeros(N_E)
for i in range(N_E):
    prob_Ef[i] = np.sum(np.abs(Phi[i, :])**2) *domega

prob_Ef /= np.sum(prob_Ef)


final_width = width_calculation(prob_Ef, Ef_vec)



final_width_eV = final_width / e

# print(f"Final energy width: {final_width_eV:.3f} eV ({final_width/E0:.3f} E0)")

# plt.figure(figsize=(8,6))
# plt.plot(Ef_vec_eV, prob_Ef, label='Final Electron Wavefunction')  # Use Ef_vec_eV instead of Ef_vec
# plt.xlabel(r'$E_f$ (eV)')  # Change label to E_f since this is final energy
# plt.ylabel(r'Probability density')
# plt.title('Final Electron State, L = %.2f [mm], width = %.2f eV (%.3f E0)'  % (L*1e3, final_width_eV, final_width/E0))  # Update title and use final_width_eV
# plt.grid(True)
# plt.legend()
# plt.show()
#%%
print("Computing photon distribution...")

# Calculate photon wavefunction by summing over final electron energies
photon_wavefunction = np.sum(np.abs(Phi)**2, axis=0)
if np.sum(photon_wavefunction) > 0:
    photon_wavefunction /= np.sum(photon_wavefunction)   # normalize

# Convert photon frequencies to energies in eV
omega_vec_eV = hbar * omega_vec / e  # Convert frequencies to photon energies in eV

# Plot photon energy distribution
plt.figure(figsize=(8,6))
plt.plot(omega_vec_eV, photon_wavefunction, 'r-', linewidth=2, label='Photon Energy Distribution')
plt.xlabel(r'Photon Energy (eV)')
plt.ylabel(r'Probability density')
plt.title('Photon Energy Distribution')
plt.grid(True)
plt.legend()
plt.show()

# Plot electron initial and final distributions comparison
plt.figure(figsize=(8,6))
plt.plot(Ef_vec_eV, psi_i*np.conj(psi_i), 'b-', linewidth=2, label='Initial Electron State')
plt.plot(Ef_vec_eV, prob_Ef, 'r-', linewidth=2, label='Final Electron State')
plt.xlabel(r'Energy (eV)')
plt.ylabel(r'Probability density')
plt.title('Electron Energy Distributions: Initial vs Final')
plt.grid(True)
plt.legend()
plt.show()
# Find peak positions in initial and final distributions
initial_peak_idx = np.argmax(psi_i_squared)
final_peak_idx = np.argmax(prob_Ef)
initial_peak_energy = Ef_vec_eV[initial_peak_idx]
final_peak_energy = Ef_vec_eV[final_peak_idx]
peak_difference = final_peak_energy - initial_peak_energy

print(f"Peak energy (initial): {initial_peak_energy:.4f} eV")
print(f"Peak energy (final): {final_peak_energy:.4f} eV")
print(f"Difference between peaks: {peak_difference:.4f} eV")
#%% Plotting
plt.figure(figsize=(12, 8))

# Plot initial distribution
plt.subplot(2, 2, 1)
plt.plot(Ef_vec, psi_i_squared, 'b-', linewidth=2, label='Initial')
plt.xlabel('Energy (eV)')
plt.ylabel('|ψᵢ(E)|²')
plt.title(f'Initial Distribution\nWidth = {initial_width/e:.2e} eV')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot final distribution
plt.subplot(2, 2, 2)
plt.plot(Ef_vec/e, rho_f, 'r-', linewidth=2, label='Final')
plt.xlabel('Energy (eV)')
plt.ylabel('ρ(Eₓ)')
plt.title(f'Final Distribution\nWidth = {final_width/e:.2e} eV')
plt.grid(True, alpha=0.3)
plt.legend()

# Compare both distributions
plt.subplot(2, 2, 3)
plt.plot(Ef_vec/e, psi_i_squared/np.max(psi_i_squared), 'b-', linewidth=2, label='Initial (normalized)')
plt.plot(Ef_vec/e, rho_f/np.max(rho_f), 'r-', linewidth=2, label='Final (normalized)')
plt.xlabel('Energy (eV)')
plt.ylabel('Normalized Probability')
plt.title('Comparison of Distributions')
plt.grid(True, alpha=0.3)
plt.legend()
#%%
# Width vs interaction length (quick scan)
plt.subplot(2, 2, 4)
L_vec = np.linspace(0.5e-3, 10e-3, 15)  # 0.5mm to 10mm
widths = []

for L_test in L_vec:
    # Quick calculation with coarser grid
    rho_test = np.zeros(N_E//2)
    Ef_coarse = Ef_vec[::2]
    
    for i, Ef in enumerate(Ef_coarse):
        integrand_vals = []
        for omega in omega_vec[::3]:  # coarser omega grid
            Ei = Ef + hbar * omega
            if Ei < 0.5*E0 or Ei > 2*E0:
                integrand_vals.append(0.0)
                continue
            
            phase_mismatch = k(Ei) - k(Ef) - q(omega)
            sinc_arg = phase_mismatch * L_test / 2
            if np.abs(sinc_arg) < 1e-10:  # Avoid division by zero
                sinc_val = 1.0
            else:
                sinc_val = np.sinc(sinc_arg / np.pi)
            integrand = (sinc_val**2) * (psi_i(Ei)**2)
            integrand_vals.append(integrand)
        
        if len(integrand_vals) > 0:
            integral_result = np.trapz(integrand_vals, omega_vec[::3])
            prefactor = (e**2 * hbar * L_test**2) / (2 * omega0 * eps0 * c) * v(E0)
            rho_test[i] = prefactor * integral_result
    
    if np.sum(rho_test) > 0:
        rho_test = rho_test / np.trapz(rho_test, Ef_coarse)
        width_test = width_calculation(rho_test, Ef_coarse)
        widths.append(width_test/e)  # Convert to eV
    else:
        widths.append(initial_width/e)  # Convert to eV

plt.plot(L_vec*1000, widths, 'ko-', markersize=4)
plt.axhline(y=initial_width/e, color='b', linestyle='--', alpha=0.7, label='Initial width')
plt.xlabel('Interaction Length (mm)')
plt.ylabel('Final Width (eV)')
plt.title('Width vs Interaction Length')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*50)
print("SIMULATION SUMMARY")
print("="*50)
print(f"Initial electron energy: {E0/e:.2e} eV")
print(f"Energy spread (ΔE): {deltaE/e:.2e} eV ({deltaE/E0:.1%} of E₀)")
print(f"Interaction length: {L*1000:.1f} mm")
print(f"Photon frequency: {omega0:.2e} rad/s")
print(f"Initial width: {initial_width/e:.4e} eV ({initial_width/E0:.4f} E₀)")
print(f"Final width: {final_width/e:.4e} eV ({final_width/E0:.4f} E₀)") 
print(f"Width difference: {(final_width - initial_width)/e:.4e} eV ({(final_width - initial_width)/E0:.4f} E₀)")
print(f"Relative change: {((final_width - initial_width)/initial_width)*100:.2f}%")