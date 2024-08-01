import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tubule_analysis_plots import create_dataframe, get_per_pixel_correlation_over_sequence

def plot_boxplot(data, x, y, palette, ylim=None, xlim=None, yticks_fontsize=12, xticks_fontsize=12, xlabel='', ylabel='', title='', filename='', figsize=(2, 6), rotation=90):
    """
    Plot a boxplot with the given parameters.

    Args:
        data (DataFrame): Data for plotting.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        palette (dict): Color palette for the plot.
        ylim (tuple, optional): Limits for y-axis.
        xlim (tuple, optional): Limits for x-axis.
        yticks_fontsize (int, optional): Font size for y-axis ticks.
        xticks_fontsize (int, optional): Font size for x-axis ticks.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        filename (str, optional): Filename to save the plot.
        figsize (tuple, optional): Size of the figure.
        rotation (int, optional): Rotation angle for x-axis ticks.
    """
    ax = sns.boxplot(data=data, x=x, y=y, showfliers=False, width=0.95, palette=palette)
    
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)

    yt = [f'{tick:.2f}' for tick in ax.get_yticks()]
    ax.set_yticklabels(yt, fontsize=yticks_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=xticks_fontsize, rotation=rotation)

    plt.title(title, fontsize=15)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.gcf().set_size_inches(*figsize)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_data(df, x_col, y_col, palette, ylim=None, figsize=(10, 6), xlabel='', ylabel='', title='', filename='', rotation=90):
    """
    Plot data using a boxplot with the given parameters.

    Args:
        df (DataFrame): Data for plotting.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        palette (dict): Color palette for the plot.
        ylim (tuple, optional): Limits for y-axis.
        figsize (tuple, optional): Size of the figure.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        filename (str, optional): Filename to save the plot.
        rotation (int, optional): Rotation angle for x-axis ticks.
    """
    ax = sns.boxplot(data=df, x=x_col, y=y_col, palette=palette)
    
    if ylim:
        ax.set_ylim(*ylim)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=24)
    plt.grid(True)
    plt.gcf().set_size_inches(*figsize)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_per_pixel_variation_over_sequence(connection, channel, variation):
    """
    Plot per-pixel variation over sequence for given connection and channel.

    Args:
        connection (str): Connection type.
        channel (str): Channel name.
        variation (str): Variation type.
    """
    groups = ['Control', 'ATL', 'Climp', 'RTN'] if channel == 'egfp' else ['ATL', 'Climp', 'RTN']
    df = create_dataframe(groups, connection, channel, variation, data_type='variation')
    colors = sns.color_palette(n_colors=4)
    palette = {'Reticulon': colors[1], 'Climp': colors[2], 'Atlastin': colors[3]}
    plot_data(df, 'Group', 'Per-pixel-variation', palette, (0, 0.13), (13, 13), 'Group', f'{variation} over sequence', f'{variation} over sequence', f'Seq_{variation}_Per_pixel_{connection}_{channel}_v4.png')

def plot_per_pixel_correlation_over_sequence(connection):
    """
    Plot per-pixel correlation over sequence for given connection.

    Args:
        connection (str): Connection type.
    """
    groups = ['ATL', 'Climp', 'RTN']
    data = []
    labels = []
    for group in groups:
        group_data = get_per_pixel_correlation_over_sequence(group, connection)
        data.extend(group_data)
        labels.extend([group] * len(group_data))
    
    df = pd.DataFrame({'Per-pixel-corr': data, 'Group': labels})
    ax = sns.violinplot(data=df, x='Group', y='Per-pixel-corr')
    ax.set_ylim(-0.4, 0.85)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    yt = [f'{tick:.2f}' for tick in ax.get_yticks()]
    ax.set_yticklabels(yt, fontsize=18)
    plt.title(f'Per pixel cross-correlation over the sequence in {connection} tubules', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Cross-correlation value', fontsize=24)
    plt.gcf().set_size_inches(10, 6)
    plt.show()