# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def compute_FWHM(x, y):
    half = np.max(y) / 2.0
    above = np.where(y >= half)[0]
    if len(above) < 2:
        return 0.0
    return x[above[-1]] - x[above[0]]


# %% 1) Physical constants
c = 2.99792458e8  # m/s
m = 9.109383e-31  # kg
e = 1.60217662e-19  # C
hbar = 1.054571800e-34  # J·s
eps0 = 8.854187817e-12  # F/m


# %%
def final_state_probability_density(
    initial_width, L_int, gamma, recoil, v_g, v_0, omega_0
):

    Delta_PM = (
        k(E_f + δE_f_grid + hbar * δω_grid)
        - k(E_f + δE_f_grid - hbar * omega0)
        - (q0 + (δω_grid / v_g) + 0.5 * recoil * δω_grid**2)
    )
    rho_f = (
        np.sum(
            (1 / np.sqrt(2 * np.pi * deltaE**2))
            * np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * deltaE**2))
            * (np.sinc(Delta_PM * L_int / 2 / np.pi)) ** 2,
            axis=1,
        )
        * dω
    )
    return rho_f


# %%
def k_rel(E):
    return np.sqrt(2 * m * E * c**2 + E**2) / (hbar * c)


v0 = 0.1 * c  # electron carrier velocity
E0 = 0.5 * m * v0**2  # central electron energy (J)

lambda0 = 500e-9  # central wavelength (m)
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
v_g = v0  # photon group velocity (m/s)

deltaE = 0.1 * hbar * omega0  # energy spread (J)
N = 2**12
# --- ENERGY GRID
N_E = N  # number of energy points

E_min = E0 - 10 * deltaE
E_max = E0 + 10 * deltaE

E_f = np.linspace(E_min, E_max, N_E)
dE = E_f[1] - E_f[0]
energy_span = E_max - E_min

δE_f = E_f - E0

N_ω = N
omega_span = 10 * deltaE / hbar  # Narrow span around ω₀
ω_min = max(omega0 - omega_span / 2, 0 * omega0)  # Start from ω₀ - span/2
ω_max = omega0 + omega_span / 2  # End at ω₀ + span/2
ω_vec = np.linspace(ω_min, ω_max, N_ω)
dω = ω_vec[1] - ω_vec[0]

δω = ω_vec - omega0

δω_grid, δE_f_grid = np.meshgrid(δω, δE_f)
q0 = k_rel(E0) - k_rel(E0 - hbar * omega0)

recoil_vec = np.linspace(-0.01, 0.01, 2)  # Recoil vector in m/s
res = [
    [
        k_rel(E0 + hbar * δω_)
        - k_rel(E0 - hbar * omega0)
        - (q0 + (δω_ / v_g) + 0.5 * recoil * δω_**2)
        for recoil in recoil_vec
    ]
    for δω_ in δω
]
plt.figure()
plt.imshow(
    res,
    extent=[δω.min(), δω.max(), recoil_vec.min() / e, recoil_vec.max() / e],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.xlabel("")
# plt.xticks(np.arange(0.099, 0.102, 0.001), rotation=45)
plt.ylabel("")
plt.show()


# %%
v0 = 0.1 * c  # electron carrier velocity
E0 = 0.5 * m * v0**2  # central electron energy (J)

lambda0 = 500e-9  # central wavelength (m)
omega0 = 2 * np.pi * c / lambda0  # central angular frequency (rad/s)
v_g = v0  # photon group velocity (m/s)

deltaE = 0.1 * hbar * omega0  # energy spread (J)

k0 = np.sqrt(2 * m * E0) / hbar  # central momentum (1/m)
k0_m_hw = np.sqrt(2 * m * (E0 - hbar * omega0)) / hbar
q0 = k0 - k0_m_hw

# From Phase‐matching expansion (approximation) to fit LINEAR dispersion relations:
recoil = -1 / (k0 * v0**2)  # second‐order term


# Define conversion functions:
def k(E):
    return np.sqrt(2 * m * E) / hbar


lambdaDB = 2 * np.pi / k0  # de Broglie wavelength (m)
# critical length for cooling (m)
L_critical = (4 / np.pi) * lambdaDB * (E0 / deltaE) ** 2
# L_int = 0.5*L_critical
L_int = 0.01  # interaction length (m)
T = L_int / v0  # interaction time (s)

N = 2**12
# --- ENERGY GRID
N_E = N  # number of energy points

E_min = E0 - 10 * deltaE
E_max = E0 + 10 * deltaE

E_f = np.linspace(E_min, E_max, N_E)
dE = E_f[1] - E_f[0]
energy_span = E_max - E_min

δE_f = E_f - E0

N_ω = N
omega_span = 10 * deltaE / hbar  # Narrow span around ω₀
ω_min = max(omega0 - omega_span / 2, 0 * omega0)  # Start from ω₀ - span/2
ω_max = omega0 + omega_span / 2  # End at ω₀ + span/2
ω_vec = np.linspace(ω_min, ω_max, N_ω)
dω = ω_vec[1] - ω_vec[0]

δω = ω_vec - omega0

δω_grid, δE_f_grid = np.meshgrid(δω, δE_f)
Delta_PM = (
    k(E0 + δE_f_grid + hbar * δω_grid)
    - k(E0 + δE_f_grid - hbar * omega0)
    - (q0 + (δω / v_g) + 0.5 * recoil * δω**2)
)
rho_f = (
    np.sum(
        (1 / np.sqrt(2 * np.pi * deltaE**2))
        * np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / 2 / deltaE**2)
        * (np.sinc(Delta_PM * L_int / 2 / np.pi)) ** 2,
        axis=1,
    )
    * dω
)
rho_f = rho_f / np.sum(rho_f * dE)  # Normalize the final state probability density
rho_i = np.exp(-(δE_f**2) / 2 / deltaE**2) / np.sqrt(
    2 * np.pi * deltaE**2
)  # Initial state probability density
initial_width = compute_FWHM(E_f, rho_i) / e  # Initial width in eV
final_width = compute_FWHM(E_f, rho_f) / e  # Final width in eV


plt.figure()
plt.plot(E_f / e, rho_i, label="initial width= {:.4f} eV".format(initial_width))
plt.plot(E_f / e, rho_f, label="final width= {:.4f} eV".format(final_width))
plt.xlabel("Energy $E$ (eV)")
plt.ylabel("Probability density")
plt.title(f"Initial and final Electron State with L_int={L_int*1000} mm")
plt.legend()
plt.show()
# %%
v_g_num = 11  # Number of group velocities to test
# Combine and sort unique values
v_g_vec = np.unique(
    np.concatenate(
        [
            np.linspace(0.099, 0.101, v_g_num) * c,
            np.linspace(0.0999, 0.1001, int(v_g_num / 2)) * c,
        ]
    )
)
widths_vg = []

for v_g_test in v_g_vec:
    rho_f = (
        np.sum(
            (1 / np.sqrt(2 * np.pi * deltaE**2))
            * np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * deltaE**2))
            * (
                np.sinc(
                    (
                        k(E0 + δE_f_grid + hbar * δω_grid)
                        - k(E0 + δE_f_grid - hbar * omega0)
                        - (q0 + (δω_grid / v_g_test) + 0.5 * recoil * δω_grid**2)
                    )
                    * L_int
                    / 2
                    / np.pi
                )
            )
            ** 2,
            axis=1,
        )
        * dω
    )

    rho_f /= np.sum(rho_f * dE)  # Normalize the final state probability density
    widths_vg.append(compute_FWHM(E_f, rho_f) / e)  # Store final width in eV

plt.figure()
plt.plot(v_g_vec / c, widths_vg, ".-")
plt.plot(
    v_g_vec / c,
    [initial_width] * len(v_g_vec),
    label=f"Initial width = {initial_width:.4f} eV",
)
plt.xlabel("Photon Group Velocity (c)")
plt.ylabel("Final Width (eV)")
plt.title(f"Final Width vs. Photon Group Velocity with L_int={L_int*1000} mm")
plt.xticks(np.arange(0.099, 0.102, 0.001), rotation=45)
plt.legend()
plt.show()

# %% Simple Width vs. Interaction Length Scan
L_num = 11  # Number of interaction lengths to test
L_int_vec = np.linspace(0.0001, 0.001, L_num)  # m
# L_int_vec = np.unique(np.concatenate([np.linspace(0.0025, 0.04, L_num), np.linspace(0.0025 , 0.00625, 6)]))
v_g = 0.1 * c  # Fixed group velocity for this scan
Delta_PM = (
    k(E0 + δE_f_grid + hbar * δω_grid)
    - k(E0 + δE_f_grid - hbar * omega0)
    - (q0 + (δω / v_g) + 0.5 * recoil * δω**2)
)

widths_L = []
probability = []
for L_int_test in L_int_vec:
    rho_f = (
        np.sum(
            (e**2 * hbar * L_int_test**2 / (2 * (δω_grid + omega0) * m**2))
            * (1 / np.sqrt(2 * np.pi * deltaE**2))
            * np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / 2 / deltaE**2)
            * (np.sinc(Delta_PM * (L_int_test / 2) / np.pi)) ** 2,
            axis=1,
        )
        * dω
    )
    norm = np.sum(rho_f * dE)  # Normalize the final state probability density
    rho_f = rho_f / norm  # Normalize the final state probability density
    probability.append(norm)  # Store the total probability
    widths_L.append(compute_FWHM(E_f, rho_f) / e)  # Store final width in eV

plt.figure()
plt.plot(L_int_vec * 1000, widths_L, ".")
plt.plot(
    L_int_vec * 1000,
    [initial_width] * len(L_int_vec),
    label=f"Initial width = {initial_width:.4f} eV",
)
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length (mm)")
plt.legend()
plt.show()

# %%
plt.figure()
plt.plot(L_int_vec * 1000, probability, ".")
plt.ylabel("Total Probability")
plt.xlabel("Interaction Length (mm)")
plt.show()

# %% 2D plot of widths vs v_g and L_int
widths_2D = np.zeros((len(L_int_vec), len(v_g_vec)))
for i, L_int_test in enumerate(L_int_vec):
    for j, v_g_test in enumerate(v_g_vec):
        rho_f = (
            np.sum(
                (e**2 * hbar * L_int_test**2 / (2 * (δω_grid + omega0) * m**2))
                * (1 / np.sqrt(2 * np.pi * deltaE**2))
                * np.exp(-((δE_f_grid + hbar * δω_grid) ** 2) / (2 * deltaE**2))
                * (
                    np.sinc(
                        (
                            k(E0 + δE_f_grid + hbar * δω_grid)
                            - k(E0 + δE_f_grid - hbar * omega0)
                            - (q0 + (δω_grid / v_g_test) + 0.5 * recoil * δω_grid**2)
                        )
                        * L_int_test
                        / 2
                        / np.pi
                    )
                )
                ** 2,
                axis=1,
            )
            * dω
        )
        rho_f /= np.sum(rho_f * dE)  # Normalize the final state probability density
        widths_2D[i, j] = compute_FWHM(E_f, rho_f) / e  # Store final width in eV
        # print(i, j, L_int_test, v_g_test/c, widths_2D[i, j])

# %% create the 2D plot
plt.figure()
plt.imshow(
    widths_2D,
    extent=[
        v_g_vec.min() / c,
        v_g_vec.max() / c,
        L_int_vec.min() * 1000,
        L_int_vec.max() * 1000,
    ],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.colorbar(label="Final Width (eV)")
plt.xlabel("Photon Group Velocity (c)")
plt.xticks(np.arange(0.099, 0.102, 0.001), rotation=45)
plt.ylabel("Interaction Length (mm)")
plt.show()

# %%

plt.figure()
# for i in range(len(v_g_vec)):
# plt.plot(v_g_vec/c, widths_2D[i,:],'.', label=f'L_int = {L_int_vec[i]*1000:.1f} mm')
for i in range(len(v_g_vec)):
    plt.plot(L_int_vec * 1000, widths_2D[:, i])
# plt.plot(L_int_vec*1000, [initial_width]*len(v_g_vec), label=f'Initial width = {initial_width:.4f} eV')
plt.ylabel("Final Width (eV)")
plt.xlabel("Interaction Length (mm)")
plt.legend()
plt.show()
# %%
