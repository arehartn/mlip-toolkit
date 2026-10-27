import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from ase import units
from ase.io import read

# --- Import validation logic to prevent duplicate math ---
try:
    from md_sim.validation import (
        _get_distance_distribution,
        _get_angle_distribution,
        _calculate_vacf,
        _calculate_vdos_spectrum,
        _get_partial_rdfs,
        _get_triplet_angle_distribution,
        traj_dt_params,
        snapshot_traj_paths,
    )
except ImportError:
    # Fallback in case the script is run directly in the same folder
    from validation import (
        _get_distance_distribution,
        _get_angle_distribution,
        _calculate_vacf,
        _calculate_vdos_spectrum,
        _get_partial_rdfs,
        _get_triplet_angle_distribution,
        traj_dt_params,
        snapshot_traj_paths,
    )

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

    if cfg.get("plot_thermo", True):
        _plot_thermo(cfg, out_dir)

    trajs_to_plot = _build_structural_traj_dict(cfg)
    burn_in = cfg.get("validation_burn_in_frames", 800)
    ref_traj = cfg.get("reference_traj_file")
    curr_traj = cfg.get("trajectory_file")

    if cfg.get("plot_structural_overlays", True) and trajs_to_plot:
        _plot_structural_overlay(trajs_to_plot, out_dir / "rdf_overlay.png", mode="RDF", burn_in=burn_in, cfg=cfg)
        _plot_structural_overlay(trajs_to_plot, out_dir / "adf_overlay.png", mode="ADF", burn_in=burn_in, cfg=cfg)
        _plot_partial_overlays(trajs_to_plot, out_dir, burn_in=burn_in, cfg=cfg)

    if cfg.get("plot_parity", True):
        snap_ref, snap_mlip = snapshot_traj_paths(cfg)
        if snap_ref.exists() and snap_mlip.exists():
            # MLIP single points on the reference frames: a true parity comparison
            plot_parity(snap_ref, snap_mlip, str(out_dir / "accuracy_parity"), cfg=cfg)
        elif ref_traj and Path(ref_traj).exists() and curr_traj and Path(curr_traj).exists():
            print("Warning: parity plot pairs frame i of two independent trajectories, "
                  "which are different configurations. Set run_snapshot_errors: true "
                  "for a true parity plot.")
            plot_parity(ref_traj, curr_traj, str(out_dir / "accuracy_parity"), cfg=cfg)
        elif ref_traj:
            print(f"Skipping parity plots. Ensure both trajectories exist.\n"
                  f"  Ref:  {ref_traj}\n  Pred: {curr_traj}")

    if cfg.get("plot_velocity_histogram", True):
        atoms_csv = cfg.get("atoms_csv")
        if atoms_csv and Path(atoms_csv).exists():
            _plot_velocity_histogram(atoms_csv, out_dir / "velocity_histogram.png")

    if trajs_to_plot:
        if cfg.get("plot_vacf_overlay", True):
            _plot_dynamical_overlay(
                trajs_to_plot, out_dir / "vacf_overlay.png",
                mode="VACF", burn_in=burn_in, cfg=cfg,
            )
        if cfg.get("plot_vdos_overlay", True):
            _plot_dynamical_overlay(
                trajs_to_plot, out_dir / "vdos_overlay.png",
                mode="VDOS", burn_in=burn_in, cfg=cfg,
            )

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


def _plot_partial_overlays(traj_dict, out_dir, burn_in=800, cfg=None):
    """Partial RDFs (cfg['partial_rdf_pairs']) and triplet ADFs (cfg['partial_adf_triplets'])."""
    if cfg is None:
        cfg = {}
    pairs = [tuple(p) for p in cfg.get("partial_rdf_pairs") or []]
    triplets = [tuple(t) for t in cfg.get("partial_adf_triplets") or []]
    if not pairs and not triplets:
        return

    lw = cfg.get("plot_style", {}).get("line_width", 2)
    rmax = cfg.get("partial_rdf_rmax", 6.0)
    rdf_bins = cfg.get("partial_rdf_bins", 100)
    adf_bins = cfg.get("partial_adf_bins", 90)
    default_rcut = cfg.get("partial_adf_rcut", 1.5)

    panels = [("RDF", p) for p in pairs] + [("ADF", t) for t in triplets]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
    axes = axes[0]

    for label, path in traj_dict.items():
        print(f"Calculating partial RDF/ADF plots for {label}...")
        try:
            full_traj = read(str(path), index=":")
            actual_burn = burn_in if burn_in < len(full_traj) else max(0, len(full_traj) // 2)
            frames = full_traj[actual_burn::10]
            if not frames:
                continue
            symbols = set(frames[0].get_chemical_symbols())
            is_ref = "Reference" in label or "AIMD" in label

            valid_pairs = [p for p in pairs if set(p) <= symbols]
            prdf, r = _get_partial_rdfs(frames, valid_pairs, rmax=rmax, bins=rdf_bins) if valid_pairs else ({}, None)

            for ax, (kind, spec) in zip(axes, panels):
                if kind == "RDF":
                    if spec not in prdf:
                        continue
                    x, y = r, prdf[spec][0]
                else:
                    if not set(spec[:3]) <= symbols:
                        continue
                    rcut = float(spec[3]) if len(spec) > 3 else default_rcut
                    y, _, x = _get_triplet_angle_distribution(frames, spec[:3], rcut, adf_bins)
                ax.plot(x, y, label=label, linewidth=lw, alpha=0.85)
                if is_ref:
                    ax.fill_between(x, y, alpha=0.2)
        except Exception as e:
            print(f"Error processing {label}: {e}")

    for ax, (kind, spec) in zip(axes, panels):
        if kind == "RDF":
            ax.set_title(f"g(r) {spec[0]}–{spec[1]}")
            ax.set_xlabel("Distance (Å)")
            ax.set_ylabel("g(r)")
            ax.set_xlim(0, rmax)
        else:
            ax.set_title(f"ADF {spec[0]}–{spec[1]}–{spec[2]}")
            ax.set_xlabel("Angle (Degrees)")
            ax.set_ylabel("Probability")
            ax.set_xlim(0, 180)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    save_path = out_dir / "partial_rdf_adf_overlay.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_dynamical_overlay(traj_dict, save_path, mode="VACF", burn_in=0, cfg=None):
    """Overlay VACF or VDOS for multiple trajectories (e.g. AIMD vs MLIP)."""
    if cfg is None:
        cfg = {}
    plt.figure(figsize=(8, 5))
    lw = cfg.get("plot_style", {}).get("line_width", 2)
    vdos_scale = cfg.get("plot_vdos_scale", "spectrum")
    if cfg.get("plot_vdos_normalized", False):
        vdos_scale = "area"

    if mode == "VACF":
        xlabel, ylabel, title = "Time (fs)", r"$C_v(t)$", "Velocity Autocorrelation Function"
        vacf_max_fs = cfg.get("plot_vacf_max_fs")
    else:
        xlabel = "Frequency (THz)"
        title = r"Vibrational DOS $g(\omega)=\int C(t)\cos(\omega t)\,dt$"
        if vdos_scale == "area":
            ylabel = r"$g(\omega)$ (shape PMF)"
        elif vdos_scale == "peak":
            ylabel = r"$g(\omega) / \max g(\omega)$"
        else:
            ylabel = r"$g(\omega)$ (a.u.)"
        vdos_xmax = cfg.get("plot_vdos_xmax", 100)

    for label, path in traj_dict.items():
        print(f"Calculating {mode} plot for {label}...")
        try:
            full_traj = read(str(path), index=":")
            if not full_traj:
                continue

            actual_burn = burn_in if burn_in < len(full_traj) else max(0, len(full_traj) // 2)
            frames = full_traj[actual_burn:]
            if len(frames) < 3:
                print(f"  Skipping {label}: need at least 3 frames after burn-in.")
                continue

            dt_fs, log_int = traj_dt_params(path, cfg)

            if mode == "VACF":
                y, x = _calculate_vacf(frames, dt_fs, log_interval=log_int)
                if vacf_max_fs is not None:
                    n = int(vacf_max_fs / (dt_fs * log_int))
                    x, y = x[:n], y[:n]
            else:
                g_omega, g_pmf, x = _calculate_vdos_spectrum(frames, dt_fs, log_interval=log_int)
                if vdos_scale == "area":
                    y = g_pmf
                elif vdos_scale == "peak":
                    y = g_omega / g_omega.max() if g_omega.max() > 0 else g_omega
                else:
                    y = g_omega

            plt.plot(x, y, label=label, linewidth=lw, alpha=0.85)

            if "Reference" in label or "AIMD" in label:
                plt.fill_between(x, y, alpha=0.15)

        except Exception as e:
            print(f"Error processing {label}: {e}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if mode == "VACF":
        plt.axhline(0, color='k', linewidth=1, linestyle=':', alpha=0.5)
    else:
        plt.xlim(0, vdos_xmax)
    plt.ylim(bottom=0 if mode == "VDOS" else None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


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
    fig_fluct, ax_fluct = plt.subplots(figsize=(10, 6))
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
            # DFT and MLIP absolute energies differ by a constant; compare fluctuations
            ax_fluct.plot(df[x_col], df[p_col] - df[p_col].mean(), label=lbl,
                          linestyle=ls, alpha=alpha, linewidth=lw)
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

        ax_fluct.set_xlabel("Simulation Step")
        ax_fluct.set_ylabel(r"$E_{pot} - \langle E_{pot} \rangle$ (eV)")
        ax_fluct.set_title("Potential Energy Fluctuations (each run's mean removed)")
        ax_fluct.legend(loc='best')
        ax_fluct.grid(True, alpha=0.3)
        fig_fluct.tight_layout()
        fig_fluct.savefig(out_dir / "pot_energy_fluct_vs_step.png", dpi=300)
    plt.close(fig_pot)
    plt.close(fig_fluct)

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
        true_e, pred_e = np.array(true_e), np.array(pred_e)
        de = pred_e - true_e
        mae = np.mean(np.abs(de)) * 1000  # to meV
        rmse = np.sqrt(np.mean(de**2)) * 1000
        offset = de.mean()
        rmse_shift = np.sqrt(np.mean((de - offset)**2)) * 1000
        s_energy = cfg.get("plot_style", {}).get("scatter_size_energy", 20)

        plt.figure(figsize=(6, 6))
        plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue', edgecolors='k', s=s_energy)

        lims = [min(true_e.min(), pred_e.min()), max(true_e.max(), pred_e.max())]
        plt.plot(lims, lims, 'r--', alpha=0.7,
                 label=f'MAE: {mae:.2f} meV/atom\nRMSE: {rmse:.2f} meV/atom\n'
                       f'RMSE (offset {offset * 1000:+.1f} removed): {rmse_shift:.2f} meV/atom')

        plt.xlabel("Reference Energy (eV/atom)")
        plt.ylabel("Predicted Energy (eV/atom)")
        plt.title("Energy Parity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_energy.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_energy.png")

    # --- 2. Force Parity ---
    true_f_all, pred_f_all, sym_all = [], [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_f_all.append(_get_forces(tf).flatten())
            pred_f_all.append(_get_forces(pf).flatten())
            sym_all.append(np.repeat(tf.get_chemical_symbols(), 3))
        except ValueError as e:
            pass

    if len(true_f_all) > 0:
        true_f = np.concatenate(true_f_all)
        pred_f = np.concatenate(pred_f_all)
        symbols = np.concatenate(sym_all)
        mae_f = np.mean(np.abs(pred_f - true_f))
        rmse_f = np.sqrt(np.mean((pred_f - true_f)**2))

        s_forces = cfg.get("plot_style", {}).get("scatter_size_forces", 1)
        plt.figure(figsize=(6, 6))
        # One colour per element, so e.g. H errors are not hidden under Na/B
        for el in dict.fromkeys(symbols):
            m = symbols == el
            rmse_el = np.sqrt(np.mean((pred_f[m] - true_f[m])**2))
            plt.scatter(true_f[m], pred_f[m], alpha=0.3, s=s_forces,
                        label=f'{el} (RMSE {rmse_el:.3f})')

        lims = [min(true_f.min(), pred_f.min()), max(true_f.max(), pred_f.max())]
        plt.plot(lims, lims, 'k--', alpha=0.5,
                 label=f'All: MAE {mae_f:.3f}, RMSE {rmse_f:.3f} eV/Å')
        
        plt.xlabel("Reference Force (eV/Å)")
        plt.ylabel("Predicted Force (eV/Å)")
        plt.title("Force Component Parity")
        plt.legend(markerscale=max(1, 8 / max(s_forces, 1e-3) ** 0.5))
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
        ax.hist(df_last[col] * units.fs, bins=50, alpha=0.7, color=color, density=True, label=label)
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_title(f"Velocity Component Distributions at Step {last_step}")
    # Fix: Removed LaTeX \text to prevent Matplotlib crashing
    axes[2].set_xlabel("Velocity (Å/fs)") 
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()