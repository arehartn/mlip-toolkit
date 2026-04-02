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
    """Robustly extract energy from a frame, checking calculators, results, and info."""
    # 1. Check if it's a SinglePointCalculator or similar with stored results
    if frame.calc is not None:
        if hasattr(frame.calc, 'results'):
            if 'energy' in frame.calc.results:
                return float(frame.calc.results['energy'])
            if 'free_energy' in frame.calc.results:
                return float(frame.calc.results['free_energy'])
        
        # Try the standard method but catch the PropertyNotImplementedError
        try:
            return frame.get_potential_energy()
        except Exception:
            pass

    # 2. Fallback: Check the info dictionary (common for .extxyz files)
    for key in ['energy', 'REF_energy', 'dft_energy', 'Energy', 'E', 'free_energy']:
        if key in frame.info:
            return float(frame.info[key])

    raise ValueError(f"Could not find energy. Available info: {list(frame.info.keys())}")


def _get_forces(frame):
    """Robustly extract forces from a frame."""
    # 1. Check calculator results
    if frame.calc is not None:
        if hasattr(frame.calc, 'results') and 'forces' in frame.calc.results:
            return np.array(frame.calc.results['forces'])
        try:
            return frame.get_forces()
        except Exception:
            pass

    # 2. Fallback: Check arrays (where .extxyz usually stores force arrays)
    for key in ['forces', 'REF_forces', 'dft_forces', 'Forces', 'force']:
        if key in frame.arrays:
            return np.array(frame.arrays[key])

    raise ValueError(f"Could not find forces. Available arrays: {list(frame.arrays.keys())}")


def plot_parity(true_traj_path, pred_traj_path, output_prefix):
    """Generates energy and force parity plots comparing reference to prediction."""
    print(f"Reading trajectories for parity:\n  Ref:  {true_traj_path}\n  Pred: {pred_traj_path}")
    
    try:
        true_frames = read(str(true_traj_path), index=":")
        pred_frames = read(str(pred_traj_path), index=":")
    except Exception as e:
        print(f"ERROR reading trajectory files: {e}")
        return

    # Match frame counts
    if len(true_frames) != len(pred_frames):
        n = min(len(true_frames), len(pred_frames))
        print(f"Warning: Frame mismatch ({len(true_frames)} vs {len(pred_frames)}). Using first {n} frames.")
        true_frames, pred_frames = true_frames[:n], pred_frames[:n]

    n_atoms = len(true_frames[0])

    # --- 1. Energy Parity ---
    true_e, pred_e = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_e.append(_get_energy(tf) / n_atoms)
            pred_e.append(_get_energy(pf) / n_atoms)
        except ValueError as e:
            print(f"Skipping frame {i} for energy: {e}")

    if len(true_e) > 0:
        mae = mean_absolute_error(true_e, pred_e) * 1000 # to meV
        
        plt.figure(figsize=(6, 6))
        plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue', edgecolors='k', s=20)
        
        # Add diagonal line
        lims = [min(min(true_e), min(pred_e)), max(max(true_e), max(pred_e))]
        plt.plot(lims, lims, 'r--', alpha=0.7, label=f'MAE: {mae:.2f} meV/atom')
        
        plt.xlabel("Reference Energy (eV/atom)")
        plt.ylabel("Predicted Energy (eV/atom)")
        plt.title("Energy Parity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_energy.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_energy.png")

    # --- 2. Force Parity ---
    true_f_all, pred_f_all = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_f_all.append(_get_forces(tf).flatten())
            pred_f_all.append(_get_forces(pf).flatten())
        except ValueError as e:
            print(f"Skipping frame {i} for forces: {e}")

    if len(true_f_all) > 0:
        true_f = np.concatenate(true_f_all)
        pred_f = np.concatenate(pred_f_all)
        mae_f = mean_absolute_error(true_f, pred_f)

        plt.figure(figsize=(6, 6))
        plt.scatter(true_f, pred_f, alpha=0.1, s=1, color='darkorange')
        
        lims = [min(min(true_f), min(pred_f)), max(max(true_f), max(pred_f))]
        plt.plot(lims, lims, 'k--', alpha=0.5, label=f'MAE: {mae_f:.3f} eV/Å')
        
        plt.xlabel("Reference Force (eV/Å)")
        plt.ylabel("Predicted Force (eV/Å)")
        plt.title("Force Component Parity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_forces.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_forces.png")