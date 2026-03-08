import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# UPGRADED COLUMN NAME SEARCH PARTIES
# ==========================================
def check_columns(df, custom_list, default_list):
    """Combines custom names with the default list and searches the dataframe."""
    # Ensure custom_list is a list (even if the user passes a single string)
    if isinstance(custom_list, str): 
        custom_list = [custom_list]
    elif custom_list is None: 
        custom_list = []
    
    # Search custom names first, then fall back to defaults
    for col in custom_list + default_list:
        if col in df.columns: 
            return col
    return None

def get_x_col(df, custom_names=None):
    return check_columns(df, custom_names, ['step', 'Step', 'Time', 'time'])

def get_tot_col(df, custom_names=None):
    return check_columns(df, custom_names, ['energy_tot_eV', 'energy_eV', 'E_tot', 'Total_Energy', 'E_tot_eV'])

def get_temp_col(df, custom_names=None):
    return check_columns(df, custom_names, ['temperature_K', 'temp_sim_K', 'temp_inst_K', 'Temperature', 'T'])

def get_kin_col(df, custom_names=None):
    return check_columns(df, custom_names, ['energy_kin_eV', 'energy_kin', 'E_kin', 'Kinetic_Energy', 'KE', 'E_kin_eV'])

def get_pot_col(df, custom_names=None):
    return check_columns(df, custom_names, ['energy_pot_eV', 'energy_pot', 'E_pot', 'Potential_Energy', 'PE', 'E_pot_eV'])

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

    output_dir = Path(params.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating overlay plots...")
    
    # Load current data
    current_df = pd.read_csv(summary_file)
    
    # Load comparison data
    compare_dfs = []
    compare_input = params.get("compare_csvs", {})

    # If the user provides a Dictionary (Custom Label -> File Path)
    if isinstance(compare_input, dict):
        for custom_label, f in compare_input.items():
            path = Path(f)
            if path.exists():
                compare_dfs.append((custom_label, pd.read_csv(path)))
            else:
                print(f"Warning: Compare file {f} not found.")

    # Fallback: If the user still provides a List (uses filename as label)
    elif isinstance(compare_input, list):
        for f in compare_input:
            path = Path(f)
            if path.exists():
                compare_dfs.append((path.name, pd.read_csv(path)))
            else:
                print(f"Warning: Compare file {f} not found.")

    cust_x = params.get("custom_x_cols", [])
    cust_tot = params.get("custom_tot_cols", [])
    cust_temp = params.get("custom_temp_cols", [])
    cust_kin = params.get("custom_kin_cols", [])
    cust_pot = params.get("custom_pot_cols", [])

    # --- Extract Custom Titles and Labels ---
    x_label_text = params.get("x_label", "Step")
    current_label = params.get("current_run_label", "Current Run")

    # ---------------------------------------------------------
    # PLOT 1: TOTAL ENERGY VS STEP
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    curr_x = get_x_col(current_df, cust_x)
    curr_col = get_tot_col(current_df, cust_tot)
    
    if curr_col and curr_x in current_df.columns:
        plt.plot(current_df[curr_x], current_df[curr_col], label=current_label, linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_x = get_x_col(df, cust_x)
        comp_col = get_tot_col(df, cust_tot)
        if comp_col and comp_x in df.columns:
            plt.plot(df[comp_x], df[comp_col], label=name, alpha=0.7)

    plt.xlabel(x_label_text)
    plt.ylabel(params.get("tot_y_label", 'Total Energy (eV)'))
    plt.title(params.get("tot_title", 'Total Energy vs Step'))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_tot_vs_step.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: TEMPERATURE VS STEP
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    curr_x = get_x_col(current_df, cust_x)
    curr_col = get_temp_col(current_df, cust_temp)
    
    if curr_col and curr_x in current_df.columns:
        plt.plot(current_df[curr_x], current_df[curr_col], label=current_label, linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_x = get_x_col(df, cust_x)
        comp_col = get_temp_col(df, cust_temp)
        if comp_col and comp_x in df.columns:
            plt.plot(df[comp_x], df[comp_col], label=name, alpha=0.7)

    plt.xlabel(x_label_text)
    plt.ylabel(params.get("temp_y_label", 'Temperature (K)'))
    plt.title(params.get("temp_title", 'Temperature vs Step'))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_vs_step.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: KINETIC & POTENTIAL ENERGY (Subplots)
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    curr_x = get_x_col(current_df, cust_x)

    # --- Top Subplot: Kinetic Energy ---
    curr_kin = get_kin_col(current_df, cust_kin)
    if curr_kin and curr_x in current_df.columns:
        axs[0].plot(current_df[curr_x], current_df[curr_kin], label=current_label, linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_x = get_x_col(df, cust_x)
        comp_kin = get_kin_col(df, cust_kin)
        if comp_kin and comp_x in df.columns:
            axs[0].plot(df[comp_x], df[comp_kin], label=name, alpha=0.7)

    axs[0].set_ylabel(params.get("kin_y_label", 'Kinetic Energy (eV)'))
    axs[0].set_title(params.get("kin_pot_title", 'Kinetic & Potential Energy vs Step'))
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom Subplot: Potential Energy ---
    curr_pot = get_pot_col(current_df, cust_pot)
    if curr_pot and curr_x in current_df.columns:
        axs[1].plot(current_df[curr_x], current_df[curr_pot], label=current_label, linewidth=2, color='black')
    
    for name, df in compare_dfs:
        comp_x = get_x_col(df, cust_x)
        comp_pot = get_pot_col(df, cust_pot)
        if comp_pot and comp_x in df.columns:
            axs[1].plot(df[comp_x], df[comp_pot], label=name, alpha=0.7)

    axs[1].set_xlabel(x_label_text)
    axs[1].set_ylabel(params.get("pot_y_label", 'Potential Energy (eV)'))
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_kin_pot_vs_step.png', dpi=300)
    plt.close()

    print("✅ All plots saved successfully!")