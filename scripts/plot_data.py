import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter

def visualize_resampling_results():
    print("🔍 Loading data for forensic analysis...")
    
    # 1. Load Original and Resampled Data
    # Assuming these are the paths from your apply_smote_tomek function
    df_orig = pd.read_pickle("data/processed/test_holdout.pkl")
    X_res, y_res = joblib.load('data/processed/train_balanced.pkl')
    
    # Prep Original Training Data for comparison (mimicking your split)
    y_orig = df_orig['TARGET']
    
    # 2. Set up Plotting Environment
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- PLOT 1: Class Distribution (The "Bar" View) ---
    print("📊 Generating Class Distribution comparison...")
    dist_data = pd.DataFrame({
        'Dataset': ['Original'] * len(y_orig) + ['Resampled'] * len(y_res),
        'Status': list(y_orig) + list(y_res)
    })
    dist_data['Status'] = dist_data['Status'].map({0: 'Repaid', 1: 'Default'})

    sns.countplot(data=dist_data, x='Dataset', hue='Status', ax=axes[0], palette='viridis')
    axes[0].set_title("Class Imbalance Resolution", fontsize=15, fontweight='bold')
    axes[0].set_ylabel("Count of Applications")
    
    # Annotate counts on bars
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points')

    # --- PLOT 2: PCA Visualization (The "Synthesis" View) ---
    print("🎨 Computing PCA for high-dimensional visualization...")
    # Using a sample of data if the dataset is massive for speed
    pca = PCA(n_components=2)
    X_res_pca = pca.fit_transform(X_res.fillna(X_res.median()))
    
    pca_df = pd.DataFrame(data=X_res_pca, columns=['PC1', 'PC2'])
    pca_df['Target'] = y_res
    
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2', hue='Target', 
        ax=axes[1], alpha=0.5, palette='magma', s=15
    )
    axes[1].set_title("Feature Space via SMOTE-Tomek (PCA)", fontsize=15, fontweight='bold')
    axes[1].set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Var)")
    axes[1].set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Var)")

    plt.tight_layout()
    output_dir = 'reports/figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created missing directory: {output_dir}")
    plt.savefig('reports/figures/resampling_analysis.png', dpi=300)
    plt.show()
    print("✅ Analysis plots saved to reports/figures/resampling_analysis.png")

if __name__ == "__main__":
    visualize_resampling_results()