import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

# --- Import validation logic to prevent duplicate math ---
try:
    from md_sim.validation import _get_distance_distribution, _get_angle_distribution
except ImportError:
    # Fallback in case the script is run directly in the same folder
    from validation import _get_distance_distribution, _get_angle_distribution

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

def _apply_plot_style(cfg):
    """Applies plot style rcParams from cfg['plot_style'], with sensible defaults."""
    s = cfg.get("plot_style", {})
    plt.rcParams.update({
        'font.size':          s.get("font_size",   12),
        'axes.titlesize':     s.get("title_size",  14),
        'axes.labelsize':     s.get("label_size",  12),
        'xtick.labelsize':    s.get("font_size",   12),
        'ytick.labelsize':    s.get("font_size",   12),
        'legend.fontsize':    s.get("legend_size", 12),
        'lines.linewidth':    s.get("line_width",  1.5),
    })


def generate_plots(cfg):
    """The central entry point. Orchestrates all requested plots."""
    _apply_plot_style(cfg)

    out_dir = Path(cfg.get("output_dir", "./plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thermodynamics Plots (Temp, Kin/Pot, Tot) with comparisons
    _plot_thermo(cfg, out_dir)
    
    # 2. Structural Overlays (RDF & ADF)
    trajs_to_plot = _build_structural_traj_dict(cfg)
    burn_in = cfg.get("validation_burn_in_frames", 800)

    if trajs_to_plot:
        _plot_structural_overlay(trajs_to_plot, out_dir / "rdf_overlay.png", mode="RDF", burn_in=burn_in, cfg=cfg)
        _plot_structural_overlay(trajs_to_plot, out_dir / "adf_overlay.png", mode="ADF", burn_in=burn_in, cfg=cfg)

    # 3. Parity Plots (Energy/Force Accuracy)
    if ref_traj and Path(ref_traj).exists() and curr_traj and Path(curr_traj).exists():
        plot_parity(ref_traj, curr_traj, str(out_dir / "accuracy_parity"), cfg=cfg)
    elif ref_traj:
        print(f"Skipping parity plots. Ensure both trajectories exist.\n"
              f"  Ref:  {ref_traj}\n  Pred: {curr_traj}")
              
    # 2.5 Velocity Histograms (v_x, v_y, v_z at last step)
    atoms_csv = cfg.get("atoms_csv")
    if atoms_csv and Path(atoms_csv).exists():
        _plot_velocity_histogram(atoms_csv, out_dir / "velocity_histogram.png")

def _build_structural_traj_dict(cfg):
    """Builds the {label: path} dict for RDF/ADF plots, using the same label keys as the rest of the plots."""
    trajs = {}

    curr_traj = cfg.get("trajectory_file")
    if curr_traj and Path(curr_traj).exists():
        label = cfg.get("current_run_label", "Current Run")
        trajs[label] = curr_traj

    ref_traj = cfg.get("reference_traj_file")
    if ref_traj and Path(ref_traj).exists():
        # Mirror the first compare_csvs key if available, otherwise fall back to a generic name
        compare_csvs = cfg.get("compare_csvs", {})
        if isinstance(compare_csvs, dict) and compare_csvs:
            ref_label = next(iter(compare_csvs))
        else:
            ref_label = cfg.get("reference_traj_label", "Reference")
        trajs[ref_label] = ref_traj

    return trajs


def plot_rdf(cfg):
    """Standalone entry point to regenerate only the RDF overlay plot."""
    _apply_plot_style(cfg)
    out_dir = Path(cfg.get("output_dir", "./plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs_to_plot = _build_structural_traj_dict(cfg)
    burn_in = cfg.get("validation_burn_in_frames", 800)
    if trajs_to_plot:
        _plot_structural_overlay(trajs_to_plot, out_dir / "rdf_overlay.png", mode="RDF", burn_in=burn_in, cfg=cfg)
        print(f"RDF plot saved to {out_dir / 'rdf_overlay.png'}")
    else:
        print("No trajectories found; nothing to plot.")


def _plot_structural_overlay(traj_dict, save_path, mode="RDF", burn_in=800, cfg=None):
    """Uses validation module logic to overlay RDF or ADF for multiple trajectories."""
    if cfg is None:
        cfg = {}
    plt.figure(figsize=(8, 5))
    
    if mode == "RDF":
        rmax, bins = 6.0, 100
        xlabel, title = "Distance (Å)", "Radial Distribution Function g(r)"
        ylabel = "g(r)"
    else:
        rcut, bins = 3.0, 90
        xlabel, title = "Angle (Degrees)", "Angular Distribution Function (ADF)"
        ylabel = "Probability Density"

    for label, path in traj_dict.items():
        print(f"Calculating {mode} plot for {label}...")
        try:
            full_traj = read(str(path), index=":")
            if not full_traj:
                continue

            actual_burn = burn_in if burn_in < len(full_traj) else max(0, len(full_traj) // 2)
            frames = full_traj[actual_burn::10]

            if not frames:
                continue

            if mode == "RDF":
                y, _, centers = _get_distance_distribution(frames, rmax=rmax, bins=bins)
            else:
                y, _, centers = _get_angle_distribution(frames, rcut=rcut, bins=bins)

            lw = cfg.get("plot_style", {}).get("line_width", 2)
            plt.plot(centers, y, label=label, linewidth=lw, alpha=0.8)

            if "Reference" in label or "AIMD" in label:
                plt.fill_between(centers, y, alpha=0.2)

        except Exception as e:
            print(f"Error processing {label}: {e}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (Steps {burn_in}+)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if mode == "RDF":
        plt.xlim(0, rmax)
        ref_lw = cfg.get("plot_style", {}).get("line_width", 2) * 0.6
        plt.axhline(1.0, color='k', linestyle='--', linewidth=ref_lw, alpha=0.5, label='g(r)=1')
    else:
        plt.xlim(0, 180)
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _load_data_sources(cfg):
    sources = []
    lw = cfg.get("plot_style", {}).get("line_width", 2.0)

    # 1. Load the primary MACE data
    main_csv = cfg.get("summary_csv")
    if main_csv and Path(main_csv).exists():
        df_main = pd.read_csv(main_csv)
        print(f"\n✅ Loaded Main MACE CSV: {main_csv} ({len(df_main)} rows)")
        sources.append({
            "df": df_main,
            "label": cfg.get("current_run_label", "Current Run"),
            "linestyle": "-",
            "alpha": 0.7,
            "linewidth": lw
        })
    elif main_csv:
        print(f"\n🚨 ERROR: Main CSV not found at '{main_csv}'")
        print(f"   -> Currently looking inside directory: {Path.cwd()}")

    # 2. Load the comparisons
    comparisons = cfg.get("compare_csvs")
    
    if not comparisons:
        return sources

    # Safely handle if comparisons is a List or a Dictionary
    comp_items = comparisons.items() if isinstance(comparisons, dict) else [(Path(p).name, p) for p in comparisons]

    for label, path in comp_items:
        if Path(path).exists():
            df_comp = pd.read_csv(path)
            print(f"✅ Loaded Comparison CSV: {path} ({len(df_comp)} rows)")
            sources.append({
                "df": df_comp,
                "label": label,
                "linestyle": "-",
                "alpha": 0.7,
                "linewidth": lw
            })
        else:
            print(f"🚨 CRITICAL ERROR: Could not find comparison file '{path}'")
            print(f"   -> Currently looking inside directory: {Path.cwd()}")
                
    return sources

def _plot_thermo(cfg, out_dir):
    """Plots separated thermodynamic properties vs step, handling different column names."""
    sources = _load_data_sources(cfg)
    if not sources:
        print("Warning: No valid CSVs found for thermo plots.")
        return

    # Create figures for Temp, Total Energy, Potential Energy, and Kinetic Energy
    fig_temp, ax_temp = plt.subplots(figsize=(10, 6))
    fig_tot, ax_tot = plt.subplots(figsize=(10, 6))
    fig_pot, ax_pot = plt.subplots(figsize=(10, 6))
    fig_kin, ax_kin = plt.subplots(figsize=(10, 6))

    plotted_temp = plotted_tot = plotted_pot = plotted_kin = False

    x_custom = cfg.get("custom_x_cols", [])
    temp_custom = cfg.get("custom_temp_cols", [])
    tot_custom = cfg.get("custom_tot_cols", [])
    kin_custom = cfg.get("custom_kin_cols", [])
    pot_custom = cfg.get("custom_pot_cols", [])

    for s in sources:
        df = s["df"]
        lbl = s["label"]
        ls = s["linestyle"]
        alpha = s["alpha"]
        lw = s["linewidth"]

        print(f"\n--- Plotting lines for: {lbl} ---")
        
        # 1. Find X column
        x_col = get_col(df, x_custom, ['step', 'Step', 'Time', 'time', 'time_ps'])
        if not x_col:
            print(f"❌ Skipping {lbl}: Could not find X-axis column! Available: {list(df.columns)}")
            continue
        print(f"  -> X-Axis: '{x_col}' (Min: {df[x_col].min()}, Max: {df[x_col].max()})")

        # 2. Temp vs Step
        t_col = get_col(df, temp_custom, ['temperature_K', 'temp_sim_K', 'temp_inst_K', 'Temperature', 'T'])
        if t_col:
            print(f"  -> Y-Axis (Temp): '{t_col}' (Min: {df[t_col].min():.2f}, Max: {df[t_col].max():.2f})")
            ax_temp.plot(df[x_col], df[t_col], label=lbl, linestyle=ls, alpha=alpha, linewidth=lw)
            plotted_temp = True
        else:
            print(f"  ⚠️ Missing Temp column! Available: {list(df.columns)}")

        # 3. Total Energy vs Step
        tot_col = get_col(df, tot_custom, ['energy_tot_eV', 'energy_eV', 'E_tot', 'Total_Energy', 'E_tot_eV', 'etot'])
        if tot_col:
            print(f"  -> Y-Axis (Tot Energy): '{tot_col}' (Min: {df[tot_col].min():.2f}, Max: {df[tot_col].max():.2f})")
            ax_tot.plot(df[x_col], df[tot_col], label=lbl, linestyle=ls, alpha=alpha, linewidth=lw)
            plotted_tot = True
        else:
            print(f"  ⚠️ Missing Total Energy column! Available: {list(df.columns)}")

        # 4. Kinetic & Potential Energy
        k_col = get_col(df, kin_custom, ['energy_kin_eV', 'energy_kin', 'E_kin', 'Kinetic_Energy', 'KE', 'E_kin_eV', 'ekin'])
        p_col = get_col(df, pot_custom, ['energy_pot_eV', 'energy_pot', 'E_pot', 'Potential_Energy', 'PE', 'E_pot_eV', 'epot'])
        
        if k_col:
            ax_kin.plot(df[x_col], df[k_col], label=lbl, linestyle=ls, alpha=alpha, linewidth=lw)
            plotted_kin = True
        if p_col:
            ax_pot.plot(df[x_col], df[p_col], label=lbl, linestyle=ls, alpha=alpha, linewidth=lw)
            plotted_pot = True

    # Save Temp Plot
    if plotted_temp:
        ax_temp.set_xlabel("Simulation Step")
        ax_temp.set_ylabel("Temperature (K)")
        ax_temp.set_title("Temperature vs. Simulation Step")
        ax_temp.legend()
        ax_temp.grid(True, alpha=0.3)
        fig_temp.tight_layout()
        fig_temp.savefig(out_dir / "temp_vs_step.png", dpi=300)
    plt.close(fig_temp)

    # Save Tot Plot
    if plotted_tot:
        ax_tot.set_xlabel("Simulation Step")
        ax_tot.set_ylabel("Total Energy (eV)")
        ax_tot.set_title("Total Energy vs. Simulation Step")
        ax_tot.legend()
        ax_tot.grid(True, alpha=0.3)
        fig_tot.tight_layout()
        fig_tot.savefig(out_dir / "tot_energy_vs_step.png", dpi=300)
    plt.close(fig_tot)

    # Save Potential Energy Plot
    if plotted_pot:
        ax_pot.set_xlabel("Simulation Step")
        ax_pot.set_ylabel("Potential Energy (eV)")
        ax_pot.set_title("Potential Energy vs. Simulation Step")
        ax_pot.legend(loc='best')
        ax_pot.grid(True, alpha=0.3)
        fig_pot.tight_layout()
        fig_pot.savefig(out_dir / "pot_energy_vs_step.png", dpi=300)
    plt.close(fig_pot)

    # Save Kinetic Energy Plot
    if plotted_kin:
        ax_kin.set_xlabel("Simulation Step")
        ax_kin.set_ylabel("Kinetic Energy (eV)")
        ax_kin.set_title("Kinetic Energy vs. Simulation Step")
        ax_kin.legend(loc='best')
        ax_kin.grid(True, alpha=0.3)
        fig_kin.tight_layout()
        fig_kin.savefig(out_dir / "kin_energy_vs_step.png", dpi=300)
    plt.close(fig_kin)

def _get_energy(frame):
    """Robustly extract energy from a frame, checking calculators, results, and info."""
    if frame.calc is not None:
        if hasattr(frame.calc, 'results'):
            if 'energy' in frame.calc.results:
                return float(frame.calc.results['energy'])
            if 'free_energy' in frame.calc.results:
                return float(frame.calc.results['free_energy'])
        try:
            return frame.get_potential_energy()
        except Exception:
            pass

    for key in ['energy', 'REF_energy', 'dft_energy', 'Energy', 'E', 'free_energy']:
        if key in frame.info:
            return float(frame.info[key])

    raise ValueError(f"Could not find energy. Available info: {list(frame.info.keys())}")


def _get_forces(frame):
    """Robustly extract forces from a frame."""
    if frame.calc is not None:
        if hasattr(frame.calc, 'results') and 'forces' in frame.calc.results:
            return np.array(frame.calc.results['forces'])
        try:
            return frame.get_forces()
        except Exception:
            pass

    for key in ['forces', 'REF_forces', 'dft_forces', 'Forces', 'force']:
        if key in frame.arrays:
            return np.array(frame.arrays[key])

    raise ValueError(f"Could not find forces. Available arrays: {list(frame.arrays.keys())}")


def plot_parity(true_traj_path, pred_traj_path, output_prefix, cfg=None):
    """Generates energy and force parity plots comparing reference to prediction."""
    if cfg is None:
        cfg = {}
    print(f"Reading trajectories for parity:\n  Ref:  {true_traj_path}\n  Pred: {pred_traj_path}")
    
    try:
        true_frames = read(str(true_traj_path), index=":")
        pred_frames = read(str(pred_traj_path), index=":")
    except Exception as e:
        print(f"ERROR reading trajectory files: {e}")
        return

    if len(true_frames) != len(pred_frames):
        n = min(len(true_frames), len(pred_frames))
        print(f"Warning: Frame mismatch ({len(true_frames)} vs {len(pred_frames)}). Using first {n} frames.")
        true_frames, pred_frames = true_frames[:n], pred_frames[:n]

    if not true_frames:
        return

    n_atoms = len(true_frames[0])

    # --- 1. Energy Parity ---
    true_e, pred_e = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_e.append(_get_energy(tf) / n_atoms)
            pred_e.append(_get_energy(pf) / n_atoms)
        except ValueError as e:
            pass 

    if len(true_e) > 0:
        mae = mean_absolute_error(true_e, pred_e) * 1000 # to meV
        s_energy = cfg.get("plot_style", {}).get("scatter_size_energy", 20)

        plt.figure(figsize=(6, 6))
        plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue', edgecolors='k', s=s_energy)
        
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
            pass 

    if len(true_f_all) > 0:
        true_f = np.concatenate(true_f_all)
        pred_f = np.concatenate(pred_f_all)
        mae_f = mean_absolute_error(true_f, pred_f)

        s_forces = cfg.get("plot_style", {}).get("scatter_size_forces", 1)
        plt.figure(figsize=(6, 6))
        plt.scatter(true_f, pred_f, alpha=0.3, s=s_forces, color='darkorange')
        
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

def _plot_velocity_histogram(atoms_csv, out_path):
    """Plots a histogram of velocities to check Maxwell-Boltzmann distribution."""
    if not Path(atoms_csv).exists():
        return
        
    df = pd.read_csv(atoms_csv)
    if not all(col in df.columns for col in ['vx', 'vy', 'vz']):
        return
        
    last_step = df['step'].max()
    df_last = df[df['step'] == last_step]
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    
    components = [
        ('vx', '$v_x$', 'royalblue'),
        ('vy', '$v_y$', 'seagreen'),
        ('vz', '$v_z$', 'indianred')
    ]
    
    for ax, (col, label, color) in zip(axes, components):
        ax.hist(df_last[col], bins=50, alpha=0.7, color=color, density=True, label=label)
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_title(f"Velocity Component Distributions at Step {last_step}")
    # Fix: Removed LaTeX \text to prevent Matplotlib crashing
    axes[2].set_xlabel("Velocity (Å/fs)") 
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()