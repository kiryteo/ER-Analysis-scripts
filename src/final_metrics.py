import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SegmentationMetrics:
    def __init__(self, analyzer, ernet, erv2, nERdy, nERdy_plus):
        self.analyzer = analyzer
        self.ernet = ernet
        self.erv2 = erv2
        self.nERdy = nERdy
        self.nERdy_plus = nERdy_plus

    def create_dataframe(self):
        """Create a DataFrame from the provided metric lists."""
        df = pd.DataFrame({
            'Metric': ['Dice score', 'F1-score', 'Jaccard Index'] * 5,
            'Value': self.analyzer + self.ernet + self.erv2 + self.nERdy + self.nERdy_plus,
            'Method': ['AnalyzER'] * 3 + ['ERnet'] * 3 + ['ERnet-v2'] * 3 + ['nERdy'] * 3 + ['nERdy+'] * 3
        })
        return df

    def plot_metrics(self, df):
        """Plot the metrics using Seaborn."""
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 12))
        ax = sns.barplot(x='Metric', y='Value', hue='Method', data=df, palette="muted")

        yt = ax.get_yticks()
        yt = [f'{y:.1f}' for y in yt]
        ax.set_yticklabels(yt, fontsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)

        plt.xlabel('Metric', fontsize=17, fontweight='bold')
        plt.ylabel('Values', fontsize=17, fontweight='bold')
        plt.title('Segmentation Performance', fontsize=19)

        plt.savefig('segmentation_metrics.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def main():
    analyzer = [0.83, 0.47, 0.31]
    ernet = [0.89, 0.69, 0.52]
    erv2 = [0.86, 0.63, 0.46]
    nERdy = [0.94, 0.81, 0.69]
    nERdy_plus = [0.96, 0.85, 0.74]

    metrics = SegmentationMetrics(analyzer, ernet, erv2, nERdy, nERdy_plus)
    df = metrics.create_dataframe()
    metrics.plot_metrics(df)

if __name__ == "__main__":
    main()