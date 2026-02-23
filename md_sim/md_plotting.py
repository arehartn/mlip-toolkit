import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# COLUMN NAME SEARCH PARTIES
# ==========================================
def get_tot_col(df):
    for col in ['energy_tot_eV', 'energy_eV', 'E_tot', 'Total_Energy', 'E_tot_eV']:
        if col in df.columns: return col
    return None

def get_temp_col(df):
    for col in ['temperature_K', 'temp_sim_K', 'temp_inst_K', 'Temperature', 'T']:
        if col in df.columns: return col
    return None

def get_kin_col(df):
    for col in ['energy_kin_eV', 'energy_kin', 'E_kin', 'Kinetic_Energy', 'KE', 'E_kin_eV']:
        if col in df.columns: return col
    return None

def get_pot_col(df):
    for col in ['energy_pot_eV', 'energy_pot', 'E_pot', 'Potential_Energy', 'PE', 'E_pot_eV']:
        if col in df.columns: return col
    return None

# ==========================================
# MAIN PLOTTING FUNCTION
# ==========================================
def generate_plots(params):
    if not params.get("make_plots", False):
        return

    summary_file = Path(params["summary_csv"])
    if not summary_file.exists():
        print(f"Warning: {summary_file} not found. Cannot generate plots.")
        return

    print("Generating overlay plots...")
    
    # Load current data
    current_df = pd.read_csv(summary_file)
    
    # Load comparison data
    compare_dfs = []
    for f in params.get("compare_csvs", []):
        path = Path(f)
        if path.exists():
            compare_dfs.append((path.name, pd.read_csv(path)))
        else:
            print(f"Warning: Compare file {f} not found.")

    # ---------------------------------------------------------
    # PLOT 1: TOTAL ENERGY VS STEP
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    curr_col = get_tot_col(current_df)
    if curr_col:
        plt.plot(current_df['step'], current_df[curr_col], label='Current Run', linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_col = get_tot_col(df)
        if comp_col:
            plt.plot(df['step'], df[comp_col], label=name, alpha=0.7)

    plt.xlabel('Step')
    plt.ylabel('Total Energy (eV)')
    plt.title('Total Energy vs Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_tot_vs_step.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: TEMPERATURE VS STEP
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    curr_col = get_temp_col(current_df)
    if curr_col:
        plt.plot(current_df['step'], current_df[curr_col], label='Current Run', linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_col = get_temp_col(df)
        if comp_col:
            plt.plot(df['step'], df[comp_col], label=name, alpha=0.7)

    plt.xlabel('Step')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temperature_vs_step.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: KINETIC & POTENTIAL ENERGY (Subplots)
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # --- Top Subplot: Kinetic Energy ---
    curr_kin = get_kin_col(current_df)
    if curr_kin:
        axs[0].plot(current_df['step'], current_df[curr_kin], label='Current Run', linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_kin = get_kin_col(df)
        if comp_kin:
            axs[0].plot(df['step'], df[comp_kin], label=name, alpha=0.7)

    axs[0].set_ylabel('Kinetic Energy (eV)')
    axs[0].set_title('Kinetic & Potential Energy vs Step')
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom Subplot: Potential Energy ---
    curr_pot = get_pot_col(current_df)
    if curr_pot:
        axs[1].plot(current_df['step'], current_df[curr_pot], label='Current Run', linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_pot = get_pot_col(df)
        if comp_pot:
            axs[1].plot(df['step'], df[comp_pot], label=name, alpha=0.7)

    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Potential Energy (eV)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('energy_kin_pot_vs_step.png', dpi=300)
    plt.close()

    print("✅ All plots saved successfully!")