import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def generate_pareto_charts(csv_path="experiments_log.csv", output_dir="experiments"):
    df = pd.read_csv(csv_path)
    
    # Filter rows with valid metrics
    df = df.dropna(subset=["INT8_Acc_%", "Flash_KB", "RAM_Arena_KB"]).copy()
    df["Alpha"] = df["Alpha"].astype(str)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Color map for Alphas
    alpha_colors = {
        '0.25': '#2b5c8f', # Navy/Blue
        '0.35': '#00a896', # Teal
        '0.5':  '#f77f00', # Amber/Orange
        '0.75': '#d62828'  # Red/Crimson
    }
    
    # Marker map for input resolution
    res_markers = {
        '32x32x1': 'o',
        '64x32x1': 's'
    }

    # Identify Pareto Frontier (Max Accuracy for given Flash Size or lower Flash Size for given Accuracy)
    # Sort by Flash_KB ascending, then INT8_Acc_% descending
    sorted_df = df.sort_values(by=["Flash_KB", "INT8_Acc_%"], ascending=[True, False])
    
    pareto_points = []
    max_acc = -1.0
    for idx, row in sorted_df.iterrows():
        if row["INT8_Acc_%"] > max_acc:
            pareto_points.append(row)
            max_acc = row["INT8_Acc_%"]
            
    pareto_df = pd.DataFrame(pareto_points)

    # Offsets específicos por experimento para evitar solapamientos
    offsets_flash = {
        'EXP_001': (0.2, -0.4),
        'EXP_002': (-0.7, -0.4),
        'EXP_003': (0.2, 0.1),
        'EXP_004': (-0.7, 0.1),
        'EXP_005': (0.2, 0.35),
        'EXP_006': (0.2, -0.5),
        'EXP_007': (0.2, -0.4),
        'EXP_008': (-0.7, 0.3),
        'EXP_009': (0.2, 0.3),
        'EXP_010': (0.2, 0.3)
    }

    # -------------------------------------------------------------
    # Plot 1: Accuracy (%) vs Flash Size (KB)
    # -------------------------------------------------------------
    for idx, row in df.iterrows():
        c = alpha_colors.get(str(row["Alpha"]), '#333333')
        m = res_markers.get(str(row["Input_Shape"]), 'o')
        ax1.scatter(row["Flash_KB"], row["INT8_Acc_%"], color=c, marker=m, s=120, zorder=4, edgecolors='black', linewidth=1)
        
        # Annotate points
        ox, oy = offsets_flash.get(row['Exp_ID'], (0.2, 0.2))
        ax1.annotate(row['Exp_ID'], (row["Flash_KB"], row["INT8_Acc_%"]), 
                     xytext=(row["Flash_KB"] + ox, row["INT8_Acc_%"] + oy),
                     fontsize=9, fontweight='bold', color='#1d3557')

    # Draw Pareto Frontier Line
    ax1.plot(pareto_df["Flash_KB"], pareto_df["INT8_Acc_%"], color='#d62828', linestyle='--', linewidth=2, label='Frontera de Pareto (Óptimo)', zorder=3)
    ax1.fill_between(pareto_df["Flash_KB"], 85, pareto_df["INT8_Acc_%"], color='#d62828', alpha=0.08)

    ax1.set_title("Frontera de Pareto: Exactitud INT8 vs Memoria Flash (KB)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Consumo de Memoria Flash (KB)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Exactitud INT8 (%)", fontsize=11, fontweight='bold')
    ax1.set_ylim(85, 98)
    ax1.set_xlim(12, 24)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # Plot 2: Macro F1-Score vs RAM Arena (KB)
    # -------------------------------------------------------------
    df_f1 = df.dropna(subset=["Macro_F1"]).copy()
    
    for idx, row in df_f1.iterrows():
        c = alpha_colors.get(str(row["Alpha"]), '#333333')
        m = res_markers.get(str(row["Input_Shape"]), 'o')
        ax2.scatter(row["RAM_Arena_KB"], row["Macro_F1"], color=c, marker=m, s=120, zorder=4, edgecolors='black', linewidth=1)
        
        ax2.annotate(row['Exp_ID'], (row["RAM_Arena_KB"], row["Macro_F1"]), 
                     xytext=(row["RAM_Arena_KB"] + 0.1, row["Macro_F1"] + 0.004),
                     fontsize=9, fontweight='bold', color='#1d3557')

    # Sort pareto for F1 vs RAM
    sorted_ram = df_f1.sort_values(by=["RAM_Arena_KB", "Macro_F1"], ascending=[True, False])
    pareto_ram = []
    max_f1 = -1.0
    for idx, row in sorted_ram.iterrows():
        if row["Macro_F1"] > max_f1:
            pareto_ram.append(row)
            max_f1 = row["Macro_F1"]
    pareto_ram_df = pd.DataFrame(pareto_ram)

    ax2.plot(pareto_ram_df["RAM_Arena_KB"], pareto_ram_df["Macro_F1"], color='#00a896', linestyle='--', linewidth=2, label='Frontera F1 vs RAM', zorder=3)
    ax2.fill_between(pareto_ram_df["RAM_Arena_KB"], 0.82, pareto_ram_df["Macro_F1"], color='#00a896', alpha=0.08)

    ax2.set_title("Equilibrio Macro F1-Score vs Memoria RAM Arena (KB)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Consumo de Memoria RAM Arena (KB)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Macro F1-Score", fontsize=11, fontweight='bold')
    ax2.set_ylim(0.82, 0.98)
    ax2.set_xlim(13, 17.5)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#d62828', lw=2, ls='--', label='Frontera de Pareto'),
        Line2D([0], [0], marker='o', color='w', label='Resolución 32x32', markerfacecolor='#555555', markersize=9),
        Line2D([0], [0], marker='s', color='w', label='Resolución 64x32', markerfacecolor='#555555', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Alpha 0.25', markerfacecolor=alpha_colors['0.25'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Alpha 0.35', markerfacecolor=alpha_colors['0.35'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Alpha 0.50', markerfacecolor=alpha_colors['0.5'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Alpha 0.75', markerfacecolor=alpha_colors['0.75'], markersize=9),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=7, frameon=True, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "pareto_tradeoff.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.savefig("pareto_tradeoff.png", bbox_inches='tight', dpi=300)
    print(f"Gráfico de Pareto generado exitosamente en: {out_file} y pareto_tradeoff.png")
    plt.close()

if __name__ == "__main__":
    generate_pareto_charts()
