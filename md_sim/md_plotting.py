import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Necessary for head-less servers/supercomputers
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

# Try to import sklearn for the MAE calculation, fall back to manual if not found
try:
    from sklearn.metrics import mean_absolute_error
except ImportError:
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# ==========================================
# 1. COLUMN SEARCH HELPERS (Your Original Logic)
# ==========================================
def check_columns(df, custom_list, default_list):
    if isinstance(custom_list, str): custom_list = [custom_list]
    elif custom_list is None: custom_list = []
    for col in custom_list + default_list:
        if col in df.columns: return col
    return None

def get_x_col(df, custom_names=None):
    return check_columns(df, custom_names, ['step', 'Step', 'Time', 'time'])

def get_tot_col(df, custom_names=None):
    return check_columns(df, custom_names, ['energy_tot_eV', 'energy_eV', 'E_tot', 'Total_Energy', 'E_tot_eV'])

def get_temp_col(df, custom_names=None):
    return check_columns(df, custom_names, ['temperature_K', 'temp_sim_K', 'temp_inst_K', 'Temperature', 'T'])

def get_kin_col(df, custom_names=None):
    return check_columns(df, custom_names, ['energy_kin_eV', 'energy_kin', 'E_kin'])

def get_pot_col(df, custom_names=None):
    return check_columns(df, custom_names, ['energy_pot_eV', 'energy_pot', 'E_pot'])

# ==========================================
# 2. TIME-SERIES PLOTS (Energy/Temp vs Step)
# ==========================================
def generate_plots(current_csv, compare_csvs=None, output_name="md_results.png"):
    """Generates the standard MD overview (Kinetic/Potential Energy & Temp)."""
    if compare_csvs is None: compare_csvs = {}
    
    current_df = pd.read_csv(current_csv)
    compare_dfs = [(name, pd.read_csv(path)) for name, path in compare_csvs.items()]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    curr_x = get_x_col(current_df)
    
    # --- Top Subplot: Kinetic & Potential Energy ---
    # Plot Kinetic
    c_kin = get_kin_col(current_df)
    if c_kin: axs[0].plot(current_df[curr_x], current_df[c_kin], label="Current Kin", color='black', alpha=0.3)
    
    # Plot Potential
    c_pot = get_pot_col(current_df)
    if c_pot: axs[0].plot(current_df[curr_x], current_df[c_pot], label="Current Pot", color='black', linewidth=2)

    for name, df in compare_dfs:
        cx, cp = get_x_col(df), get_pot_col(df)
        if cp: axs[0].plot(df[cx], df[cp], label=f"{name} Pot", linestyle='--')

    axs[0].set_ylabel("Energy (eV)")
    axs[0].set_title("Energy vs Step")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- Bottom Subplot: Temperature ---
    c_temp = get_temp_col(current_df)
    if c_temp: axs[1].plot(current_df[curr_x], current_df[c_temp], color='red', label="Temperature")
    
    axs[1].set_ylabel("Temperature (K)")
    axs[1].set_xlabel("Step")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"Saved MD Overview Plot to {output_name}")

# ==========================================
# 3. VELOCITY HISTOGRAMS (3 Subplots)
# ==========================================
def plot_velocity_histogram(atoms_csv_path="md_atoms.csv", output_png="velocity_histogram.png"):
    try:
        df = pd.read_csv(atoms_csv_path)
    except FileNotFoundError:
        print(f"Skipping velocity plot: {atoms_csv_path} not found.")
        return

    last_step = df['step'].max()
    df_last = df[df['step'] == last_step]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Velocity Distributions at Step {last_step}', fontsize=14)
    
    colors = ['red', 'green', 'blue']
    labels = ['vx', 'vy', 'vz']
    
    for i, col in enumerate(labels):
        axs[i].hist(df_last[col], bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        axs[i].set_title(f'{col.upper()} Distribution')
        axs[i].set_xlabel('Velocity')
        axs[i].set_ylabel('Atom Count')
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved Velocity Histograms to {output_png}")

# ==========================================
# 4. ACCURACY PLOTS (Parity Plots)
# ==========================================
def plot_parity(true_traj_path, pred_traj_path, output_prefix="accuracy_parity"):
    print("Loading trajectories for Accuracy (Parity) Plots...")
    try:
        true_frames = read(true_traj_path, index=':')
        pred_frames = read(pred_traj_path, index=':')
    except Exception as e:
        print(f"Error loading trajectories for parity: {e}")
        return

    # --- Energy Parity ---
    n = len(true_frames[0])
    true_e = [f.get_potential_energy()/n for f in true_frames]
    pred_e = [f.get_potential_energy()/n for f in pred_frames]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue')
    lims = [min(true_e), max(true_e)]
    plt.plot(lims, lims, 'r--', label="Perfect Agreement")
    plt.title(f"Energy Parity (MAE: {mean_absolute_error(true_e, pred_e)*1000:.2f} meV/atom)")
    plt.xlabel("True r2SCAN Energy (eV/atom)")
    plt.ylabel("Predicted PBE Energy (eV/atom)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_prefix}_energy.png", dpi=300); plt.close()

    # --- Force Parity ---
    true_f = np.concatenate([f.get_forces().flatten() for f in true_frames])
    pred_f = np.concatenate([f.get_forces().flatten() for f in pred_frames])
    
    plt.figure(figsize=(6, 6))
    plt.scatter(true_f, pred_f, alpha=0.1, s=1, color='darkorange')
    lims = [min(true_f), max(true_f)]
    plt.plot(lims, lims, 'r--')
    plt.title(f"Force Parity (MAE: {mean_absolute_error(true_f, pred_f):.3f} eV/Å)")
    plt.xlabel("True r2SCAN Force (eV/Å)")
    plt.ylabel("Predicted PBE Force (eV/Å)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_prefix}_force.png", dpi=300); plt.close()
    print(f"Saved Accuracy Plots to {output_prefix}_energy/force.png")

# ==========================================
# 5. IMPLEMENTATION / RUNNER
# ==========================================
if __name__ == "__main__":
    # A. Standard MD results
    generate_plots(current_csv="md_summary.csv", 
                   compare_csvs={"PBE_Model": "pbe_predictions.csv"})
    
    # B. Velocity Bell Curves
    plot_velocity_histogram(atoms_csv_path="md_atoms.csv")
    
    # C. Accuracy / Parity (The important ones!)
    # Assumes 'large_traj.traj' is r2SCAN and 'pbe_evaluated.traj' is your PBE evaluation
    plot_parity(true_traj_path="large_traj.traj", 
                pred_traj_path="pbe_evaluated.traj")