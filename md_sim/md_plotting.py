import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

try:
    from sklearn.metrics import mean_absolute_error
except ImportError:
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# --- Robust Column Searching ---
def get_col(df, custom_list, default_list):
    """Searches for columns in order of preference."""
    search_list = (custom_list if custom_list else []) + default_list
    for col in search_list:
        if col in df.columns: return col
    return None

def generate_plots(cfg):
    """The central entry point. Orchestrates all requested plots."""
    out_dir = Path(cfg.get("output_dir", "./plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard MD Summary (Energy/Temp vs Step)
    _plot_summary(cfg, out_dir)
    
    # 2. Velocity Histograms (v_x, v_y, v_z at last step)
    atoms_csv = cfg.get("atoms_csv")
    if atoms_csv and Path(atoms_csv).exists():
        _plot_velocity_histogram(atoms_csv, out_dir / "velocity_histogram.png")

    # 3. Parity Plots (Energy/Force Accuracy)
    ref_traj = cfg.get("reference_traj_file")
    curr_traj = cfg.get("trajectory_file")
    if ref_traj and Path(ref_traj).exists() and Path(curr_traj).exists():
        plot_parity(ref_traj, curr_traj, out_dir / "accuracy_parity")
    elif ref_traj and not Path(ref_traj).exists():
        print(f"WARNING: reference_traj_file not found: {ref_traj}")
    elif curr_traj and not Path(curr_traj).exists():
        print(f"WARNING: trajectory_file not found: {curr_traj}")

def _plot_summary(cfg, out_dir):
    current_csv = cfg.get("summary_csv")
    if not Path(current_csv).exists(): return
    
    df = pd.read_csv(current_csv)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    x_col = get_col(df, cfg.get("custom_x_cols"), ['step', 'time_ps'])
    pot_col = get_col(df, cfg.get("custom_pot_cols"), ['energy_pot_eV', 'E_pot', 'epot_eV'])
    temp_col = get_col(df, cfg.get("custom_temp_cols"), ['temperature_K', 'T'])

    if pot_col:
        axs[0].plot(df[x_col], df[pot_col], label=cfg.get("current_run_label", "Current"), color='black', linewidth=2)
    
    compare_csvs = cfg.get("compare_csvs", {})
    for label, path in compare_csvs.items():
        if Path(path).exists():
            cdf = pd.read_csv(path)
            cx = get_col(cdf, cfg.get("custom_x_cols"), ['step'])
            cp = get_col(cdf, cfg.get("custom_pot_cols"), ['energy_pot_eV', 'epot_eV'])
            if cp: axs[0].plot(cdf[cx], cdf[cp], label=label, linestyle='--', alpha=0.7)

    axs[0].set_ylabel("Potential Energy (eV)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    if temp_col:
        axs[1].plot(df[x_col], df[temp_col], color='red')
        axs[1].set_ylabel("Temperature (K)")
        axs[1].set_xlabel("Simulation Step")
    
    plt.tight_layout()
    plt.savefig(out_dir / "md_summary.png", dpi=300)
    plt.close()

def _plot_velocity_histogram(csv_path, output_png):
    df = pd.read_csv(csv_path)
    last_step = df['step'].max()
    df_last = df[df['step'] == last_step]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Velocity Distributions at Step {last_step}', fontsize=14)
    
    for i, (col, color) in enumerate(zip(['vx', 'vy', 'vz'], ['red', 'green', 'blue'])):
        axs[i].hist(df_last[col], bins=30, alpha=0.7, color=color, edgecolor='black')
        axs[i].set_title(f'{col.upper()} Distribution')
        axs[i].set_xlabel('Velocity')
        axs[i].set_ylabel('Atom Count')
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def _get_energy(frame):
    """
    Robustly extract energy from a frame regardless of source format.
    - For frames with a calculator (e.g. .traj from MACE): use get_potential_energy()
    - For extxyz/DFT reference frames: energy lives in atoms.info dict
    """
    # Try calculator first (predicted/ML frames)
    if frame.calc is not None:
        try:
            return frame.get_potential_energy()
        except Exception:
            pass

    # Fallback: extxyz stores energy in info dict under various key names
    for key in ['energy', 'REF_energy', 'dft_energy', 'Energy', 'E']:
        if key in frame.info:
            return float(frame.info[key])

    raise ValueError(
        f"Could not extract energy from frame. "
        f"Calculator: {frame.calc}, info keys: {list(frame.info.keys())}"
    )


def _get_forces(frame):
    """
    Robustly extract forces from a frame regardless of source format.
    - For frames with a calculator: use get_forces()
    - For extxyz/DFT reference frames: forces live in arrays dict
    """
    if frame.calc is not None:
        try:
            return frame.get_forces()
        except Exception:
            pass

    for key in ['forces', 'REF_forces', 'dft_forces', 'Forces', 'force']:
        if key in frame.arrays:
            return np.array(frame.arrays[key])

    raise ValueError(
        f"Could not extract forces from frame. "
        f"Calculator: {frame.calc}, arrays keys: {list(frame.arrays.keys())}"
    )


def plot_parity(true_traj_path, pred_traj_path, output_prefix):
    """Compares reference energies/forces vs ML predictions."""
    print(f"Loading reference: {true_traj_path}")
    print(f"Loading predicted: {pred_traj_path}")

    true_frames = read(true_traj_path, index=':')
    pred_frames = read(pred_traj_path, index=':')

    print(f"Reference frames: {len(true_frames)}, Predicted frames: {len(pred_frames)}")

    # Match frame counts — use the minimum to avoid index errors
    n_frames = min(len(true_frames), len(pred_frames))
    if len(true_frames) != len(pred_frames):
        print(f"WARNING: Frame count mismatch. Using first {n_frames} frames for parity.")
    true_frames = true_frames[:n_frames]
    pred_frames = pred_frames[:n_frames]

    n_atoms = len(true_frames[0])

    # --- Extract energies ---
    true_e, pred_e = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_e.append(_get_energy(tf) / n_atoms)
            pred_e.append(_get_energy(pf) / n_atoms)
        except ValueError as e:
            print(f"Skipping frame {i} for energy: {e}")

    if len(true_e) == 0:
        print("ERROR: No energy data could be extracted. Skipping energy parity plot.")
    else:
        print(f"Plotting energy parity ({len(true_e)} frames)...")
        plt.figure(figsize=(6, 6))
        plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue')
        lims = [min(min(true_e), min(pred_e)), max(max(true_e), max(pred_e))]
        plt.plot(lims, lims, 'r--', label="Perfect Agreement")
        mae = mean_absolute_error(true_e, pred_e) * 1000
        plt.title(f"Energy Parity (MAE: {mae:.2f} meV/atom)")
        plt.xlabel("Reference Energy (eV/atom)")
        plt.ylabel("Predicted Energy (eV/atom)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_energy.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_energy.png  |  MAE = {mae:.2f} meV/atom")

    # --- Extract forces ---
    true_f_all, pred_f_all = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_f_all.append(_get_forces(tf).flatten())
            pred_f_all.append(_get_forces(pf).flatten())
        except ValueError as e:
            print(f"Skipping frame {i} for forces: {e}")

    if len(true_f_all) == 0:
        print("ERROR: No force data could be extracted. Skipping force parity plot.")
    else:
        true_f = np.concatenate(true_f_all)
        pred_f = np.concatenate(pred_f_all)
        print(f"Plotting force parity ({len(true_f_all)} frames, {len(true_f)} components)...")

        plt.figure(figsize=(6, 6))
        plt.scatter(true_f, pred_f, alpha=0.1, s=1, color='darkorange')
        lims = [min(min(true_f), min(pred_f)), max(max(true_f), max(pred_f))]
        plt.plot(lims, lims, 'r--')
        mae_f = mean_absolute_error(true_f, pred_f)
        plt.title(f"Force Parity (MAE: {mae_f:.3f} eV/Å)")
        plt.xlabel("Reference Force (eV/Å)")
        plt.ylabel("Predicted Force (eV/Å)")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_force.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_force.png  |  MAE = {mae_f:.3f} eV/Å")