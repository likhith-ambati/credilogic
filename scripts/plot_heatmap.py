import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_policy_heatmap():
    print("📊 Generating CrediLogic Dynamic Threshold Heatmap...")
    
    # 1. Define the simulation range for the Federal Funds Rate
    # Simulating rates from 3.0% to 8.0% in 0.5% increments
    macro_rates = np.arange(3.0, 8.5, 0.5) 
    
    # 2. Define the core applicant segments and their base thresholds
    segments = {
        "High Stress": 0.08, 
        "Standard": 0.11, 
        "High Stability": 0.15
    }
    
    # 3. Build the data matrix applying the Macro-Adaptive Logic
    heatmap_data = []
    for rate in macro_rates:
        row = []
        for segment, base in segments.items():
            # The Core Logic: Apply 15% strictness multiplier (0.85) if rate > 5.0%
            val = round(base * 0.85, 3) if rate > 5.0 else base
            row.append(val)
        heatmap_data.append(row)

    # 4. Convert to a formatted DataFrame for Seaborn
    heatmap_df = pd.DataFrame(
        heatmap_data, 
        index=[f"{r:.1f}%" for r in macro_rates], 
        columns=segments.keys()
    )
    
    # 5. Render the Professional Visual
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap_df, 
        annot=True,          
        fmt=".3f",           
        cmap="YlGnBu",       # Professional financial color map
        cbar_kws={'label': 'Decision Threshold (Lower = Stricter)'},
        linewidths=1,
        linecolor='white'
    )
    
    # Add visual divider to explicitly show the "5.0% Trigger Line"
    # 4.5 is the index position on the y-axis between 5.0% and 5.5%
    ax.axhline(y=5, color='red', linewidth=3, linestyle='--')
    plt.text(1.5, 4.8, 'RESTRICTIVE SHIFT (Rate > 5.0%)', color='red', weight='bold', ha='center')

    # Formatting
    plt.title("CrediLogic Policy Matrix: Dynamic Thresholds vs. Macro Rates", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Applicant Risk Segment", fontsize=12, labelpad=10)
    plt.ylabel("Federal Funds Rate (%)", fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_policy_heatmap()