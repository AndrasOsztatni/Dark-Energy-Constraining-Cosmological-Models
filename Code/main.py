import matplotlib.pyplot as plt
from scipy import integrate 
import numpy as np
import pandas as pd


#-----Constants-----
H0 = 70 #km/sMpc
OMEGA_M_VALUES = np.array([0.1, 0.3, 0.6, 1])
Z_MIN = 0
Z_MAX = 2
Z_GRID = np.linspace(Z_MIN, Z_MAX, 1000)
C_LIGHT = 299792.458
OMEGA_LINSPACE = np.linspace(0.2, 1, 100)
H0_LINSPACE = np.linspace(60, 80, 100)

#-----Functions-----
def hubble_parameter(z:np.ndarray, omega_m: float, h0: float ) -> np.ndarray:
    omega_lambda = 1.0 - omega_m
    return np.sqrt(h0**2 * (omega_m * (1.0 + z)**3 + omega_lambda))

def luminosity_distance(z: np.ndarray, omega_m: float, h0: float, c: float = C_LIGHT) -> np.ndarray:
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]
    
    
    z_with_zero = np.insert(z_sorted, 0, 0.0)
    h_z = hubble_parameter(z_with_zero, omega_m, h0)
    
    
    integral_sorted = integrate.cumulative_trapezoid(1.0 / h_z, z_with_zero, initial=0.0)[1:]
    d_l_sorted = c * (1.0 + z_sorted) * integral_sorted
    
    d_l = np.empty_like(d_l_sorted)
    d_l[sort_idx] = d_l_sorted
    return d_l

def distance_modulus(d_l: np.ndarray) -> np.ndarray:
  with np.errstate(divide="ignore", invalid="ignore"):
    mu = 5.0 * np.log10(d_l) + 25.0
  mu[~np.isfinite(mu)] = np.nan
  return mu

def processing_data():
  global Union_Redshift, Distance_Modulus, Distance_Modulus_Error
  df = pd.read_csv("SCPUnion2.1_mu_vs_z.txt", delimiter="\t")
  Union_Redshift = df['Redshift'].values
  Distance_Modulus = df['Modulus'].values
  Distance_Modulus_Error = df['Mod_err'].values
  fig, ax=plt.subplots(figsize=(8,6))
  ax.errorbar(
    x=Union_Redshift,
    y=Distance_Modulus,
    yerr=Distance_Modulus_Error,
    fmt="o",
    color="tab:blue",
    ecolor="gray",
    elinewidth=1.0,
    capsize=3,
    linestyle="None",
    label="Observations"
  )
  ax.set_xlabel("Redshift ($z$)", fontsize=12)
  ax.set_ylabel("Distance Modulus ($\mu$)", fontsize=12)
  ax.set_title("Union 2.1 Dataset Redshift vs. Distance Modulus with Errors", fontsize=14)
  ax.legend(frameon=True)
  ax.grid(True)
  plt.savefig("Union2.1dataset.pdf", format="pdf")
  plt.close(fig)

def compute_chi_2(z_i: np.ndarray, mu_i: np.ndarray, sigma: np.ndarray, om: float, h0: float) -> float:
    d_l_th = luminosity_distance(z_i, om, h0)
    
    
    valid = (d_l_th > 0) & (sigma > 0) & np.isfinite(d_l_th)
    if not np.any(valid):
        return np.inf  
        
    mu_th = 5.0 * np.log10(d_l_th[valid]) + 25.0
    residuals = (mu_i[valid] - mu_th) / sigma[valid]
    
    return np.sum(residuals**2)


def main():
  plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
  
  # 1. Hubble Parameter Plot
  fig, ax = plt.subplots(figsize=(8, 6))
  for omega_m in OMEGA_M_VALUES:
    ax.plot(Z_GRID, hubble_parameter(Z_GRID, omega_m, H0), label=rf"$\Omega_m = {omega_m}$")
  
  ax.set_xlabel("Redshift ($z$)", fontsize=12)
  ax.set_ylabel("Hubble Parameter $H(z)$ [km/s/Mpc]", fontsize=12)
  ax.set_title("Hubble Parameter vs. Redshift", fontsize=14)
  ax.legend(frameon=True)
  fig.savefig("Hubble_Parameter_depending_on_Redshift.pdf", bbox_inches="tight")
  plt.close(fig)

  # 2. Luminosity Distance Plot
  fig, ax = plt.subplots(figsize=(8, 6))
  d_l_results = []
  
  for omega_m in OMEGA_M_VALUES:
    d_l = luminosity_distance(Z_GRID, omega_m, H0)
    d_l_results.append(d_l)
    ax.plot(Z_GRID, d_l, label=rf"$\Omega_m = {omega_m}$")
    
  ax.set_xlabel("Redshift ($z$)", fontsize=12)
  ax.set_ylabel("Luminosity Distance $d_L$ [Mpc]", fontsize=12)
  ax.set_title("Luminosity Distance vs. Redshift", fontsize=14)
  ax.legend(frameon=True)
  fig.savefig("Luminosity_distance.pdf", bbox_inches="tight")
  plt.close(fig)

  # 3. Distance Modulus Plot
  fig, ax = plt.subplots(figsize=(8, 6))
  for i, omega_m in enumerate(OMEGA_M_VALUES):
    mu = distance_modulus(d_l_results[i])
    ax.plot(Z_GRID, mu, label=rf"$\Omega_m = {omega_m}$")
    
  ax.set_xlabel("Redshift ($z$)", fontsize=12)
  ax.set_ylabel("Distance Modulus $\mu$", fontsize=12)
  ax.set_title("Distance Modulus vs. Redshift", fontsize=14)
  ax.legend(frameon=True)
  fig.savefig("Distance_modulus.pdf", bbox_inches="tight")
  plt.close(fig)

  #4.0 Union 2.1 Dataset
  processing_data()

  #4.1 Chi^2
  Om_grid, H0_grid = np.meshgrid(OMEGA_LINSPACE, H0_LINSPACE)
  Chi2_grid = np.zeros_like(Om_grid)
  global chi_min, om_min, h0_min
  chi_min, om_min, h0_min = 1e30, 0, 0
  for i in range(H0_LINSPACE.shape[0]):
    for j in range(OMEGA_LINSPACE.shape[0]):
      Chi2_grid[i, j] = compute_chi_2(
        Union_Redshift, 
        Distance_Modulus, 
        Distance_Modulus_Error, 
        Om_grid[i, j], 
        H0_grid[i, j]
        )
      if Chi2_grid[i, j]<chi_min:
        chi_min=Chi2_grid[i, j]
        om_min = OMEGA_LINSPACE[j]
        h0_min = H0_LINSPACE[i]
  print(om_min, h0_min)
  fig, ax = plt.subplots(figsize=(8, 6))
  contour = ax.contourf(Om_grid, H0_grid, Chi2_grid, levels=100, cmap="viridis")
  cbar = fig.colorbar(contour)
  cbar.set_label(r"$\chi^2$", fontsize=12)
  ax.set_xlabel(r"Matter Density ($\Omega_m$)", fontsize=12)
  ax.set_ylabel(r"Hubble Constant ($H_0$) [km/s/Mpc]", fontsize=12)
  ax.set_title(r"$\chi^2$ Contour Plot for Cosmological Parameters", fontsize=14)
  ax.scatter([om_min], [h0_min], marker="*")
  plt.tight_layout()
  plt.savefig("Chi2_contour_plot.pdf", format="pdf")
  plt.close(fig)

# 4.3 3D Chi^2 Surface Plot
  fig = plt.figure(figsize=(10, 7))
  ax3d = fig.add_subplot(111, projection="3d")

  surf = ax3d.plot_surface(
      Om_grid,
      H0_grid,
      Chi2_grid,
      cmap="viridis",
      edgecolor="none",
      alpha=0.85,
      antialiased=True,
      zorder=1
  )
  cbar3d = fig.colorbar(surf, ax=ax3d, shrink=0.6, pad=0.1)
  cbar3d.set_label(r"$\chi^2$", fontsize=12)

  z_offset = 0.03 * (np.max(Chi2_grid) - chi_min)
  # Mark the global minimum in 3D space
  ax3d.scatter(
      [om_min], [h0_min], [chi_min + z_offset],
      edgecolor="black", s=150, marker="*", zorder=10, depthshade=False, alpha=1.0,
      label=rf"Min $\chi^2 = {chi_min:.1f}$ ($\Omega_m={om_min:.2f}, H_0={h0_min:.1f}$)"
  )

  ax3d.set_xlabel(r"Matter Density ($\Omega_m$)", fontsize=11, labelpad=10)
  ax3d.set_ylabel(r"Hubble Constant ($H_0$) [km/s/Mpc]", fontsize=11, labelpad=10)
  ax3d.set_zlabel(r"$\chi^2$", fontsize=11, labelpad=10)
  ax3d.set_title(r"3D $\chi^2$ Surface over $(\Omega_m, H_0)$", fontsize=14)
  ax3d.view_init(elev=30, azim=-125)
  ax3d.legend(loc="upper left")

  fig.savefig("Chi2_3D_surface.pdf", format="pdf", bbox_inches="tight")
  plt.close(fig)

  #Checking Parameters
  d_l_bestfit = luminosity_distance(Z_GRID, om_min, h0_min)
  mu_bestfit = distance_modulus(d_l_bestfit)
  fig, ax = plt.subplots(figsize=(8,6))
  
  ax.errorbar(Union_Redshift, Distance_Modulus, yerr=Distance_Modulus_Error, fmt="o", color="tab:blue", ecolor="gray", elinewidth=1.0, capsize=3, linestyle="None", label="Observations", zorder=1)
  ax.plot(Z_GRID, mu_bestfit, label=rf"Best Fit: $\Omega_m = {om_min: 0.5f}, H_0 = {h0_min: 0.5f}$", color="red", zorder=10)
  ax.set_xlabel(r"Redshift $(z)$", fontsize=12)
  ax.set_ylabel(r"Distance Modulus $(\mu)$", fontsize=12)
  ax.set_title("Union 2.1 Dataset Redshift vs. Distance Modulus with Errors and Best Fit", fontsize=14)
  ax.legend(frameon=True)
  ax.grid(True)
  plt.savefig("Union2.1dataset_w_bestfit.pdf", format="pdf")
  plt.close(fig)

  print("All plots successfully generated and saved as professional PDFs!")



if __name__ == "__main__":
  main()


