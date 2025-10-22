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
from matplotlib import cm
from matplotlib.colors import Normalize

# %%  1D graph width vs v0 :
df_v0_loaded = pd.read_csv("width_vs_v0.csv")
v0_vec_loaded = df_v0_loaded["v0"].to_numpy(dtype=float)
plt.figure(figsize=(8, 5))
plt.plot(v0_vec_loaded/c, df_v0_loaded["width"]/initial_width,  linestyle='-')
plt.axhline(1, color="gray", linestyle=":")
plt.axvline(vg/c, color="red", linestyle="--", label=r"$v_0 = v_g$")
plt.xlim(v0_vec_loaded[0]/c, v0_vec_loaded[-1]/c)
# plt.yscale("log")
# plt.xlabel("Electron velocity $v_0$ (c)")
# plt.ylabel("Final width/initial width")
# plt.title(f"Final width vs $v_0$\n($L_{{int}} = {L_int:.3g}$ m = {L_int/lambda0:.2f} $\\lambda_0$)")
plt.legend()
# plt.tight_layout()
plt.savefig("width_vs_v0.svg", format="svg")
plt.show()