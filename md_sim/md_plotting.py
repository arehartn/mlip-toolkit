import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Crucial for the Ohio Supercomputer to prevent display errors
import matplotlib.pyplot as plt

def generate_plots(params):
    """
    Reads the summary CSV and generates static plots if 'make_plots' is True.
    """
    # 1. Check if the user wants to graph
    if not params.get("make_plots", False):
        return

    summary_file = params["summary_csv"]
    if not summary_file.exists():
        print(f"Warning: {summary_file} not found. Cannot generate plots.")
        return

    print("Generating plots...")
    df = pd.read_csv(summary_file)

    # Plot 1: Energy vs Time
    plt.figure(figsize=(8,5))
    plt.plot(df['time_ps'], df['energy_eV'], color='blue')
    plt.xlabel('Time (ps)')
    plt.ylabel('Energy (eV)')
    plt.title('Energy vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_vs_time.png', dpi=300)
    plt.close()

    # Plot 2: Temperature vs Time
    plt.figure(figsize=(8,5))
    plt.plot(df['time_ps'], df['temperature_K'], color='red')
    plt.xlabel('Time (ps)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temperature_vs_time.png', dpi=300)
    plt.close()
    
    # Plot 3: Energy Histogram
    plt.figure(figsize=(8,5))
    plt.hist(df['energy_eV'], bins=20, color='purple', edgecolor='black')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Frequency')
    plt.title('Energy Distribution Histogram')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('energy_histogram.png', dpi=300)
    plt.close()

    # Plot 4: Scatter plot of Temperature vs Energy
    plt.figure(figsize=(8,5))
    plt.scatter(df['energy_eV'], df['temperature_K'], color='orange', alpha=0.7)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Energy Scatter Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temperature_vs_energy_scatter.png', dpi=300)
    plt.close()

    print("✅ All plots saved as PNGs.")