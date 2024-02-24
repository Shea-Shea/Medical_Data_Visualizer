import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sys
print("Python",sys.version)
print("numpy", np.version.version)
print("Pandas", pd.__version__)


# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where(
    (df['weight'] / ((df['height']/100)**2) > 25), 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
# Where all values are not equal to one - keep those values. Set all others to 0.
df[['cholesterol', 'gluc']] = df[[
    'cholesterol', 'gluc']].where(df[['cholesterol', 'gluc']] != 1, 0)

# Where all values are less than 1 - keep those values. Set all others to 1.
df[['cholesterol', 'gluc']] = df[[
    'cholesterol', 'gluc']].where(df[['cholesterol', 'gluc']] <= 1, 1)

# Draw Categorical Plot


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(
        df[['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight', 'cardio']], id_vars='cardio')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby('cardio').value_counts().reset_index()
    #.reset_index().rename(columns={'count': 'total'})

    

    df_cat.columns = ['cardio', 'variable', 'value', 'total']
    #print(df_cat)
    #print(pd.__version__)

    #df_cat.info()
    # Draw the catplot with 'sns.catplot()'
    # Get the figure for the output
    
    fig = sns.catplot(data=df_cat, x='variable',
                      y='total', hue='value', col='cardio', order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], kind='bar', ci=None).fig

  
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df.ap_lo <= df.ap_hi) & (
        df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(.975)) & (df['weight'] >= df['weight'].quantile(.025)) & (df.weight <= df.weight.quantile(.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, linewidth=0.5,
                linecolor='white', annot=True, center=0, fmt='.1f', vmin=-0.15, vmax=0.30)
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
