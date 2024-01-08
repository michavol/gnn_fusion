import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Define a function to extract the number before "samples"
def extract_batch_size(model_name):
    match = re.search(r'(\d+)samples', model_name)
    return int(match.group(1)) if match else None

# Read the CSV file
df = pd.read_csv('results/optimization_MAE_emd_batchsize.csv')

df_ot = df[df["Model"].str.contains("ot")]
# Extract the third last character from the 'Model' column and create a new 'runs' column
df_ot['run'] = df_ot['Model'].str.slice(-11, -10)
df_ot['batch_size'] = df_ot['Model'].apply(extract_batch_size)

# Set Seaborn style to pastel
sns.set(style="whitegrid", palette="pastel")

# Create the line plot
plt.figure(figsize=(12, 6))
ax = sns.lineplot(x='batch_size', y=' MAE', data=df_ot, ci='sd', err_style="band", marker='o', label='Mean')

# Annotate mean values at each data point
for line in ax.lines:
    x_data, y_data = line.get_data()
    for i, txt in enumerate(y_data):
        mean_value = np.mean(df_ot[df_ot['batch_size'] == x_data[i]][' MAE'])
        ax.text(x_data[i], txt, f'{mean_value:.2f}', ha='center', va='bottom', color='black', fontsize=10)

# # Set titles and labels
plt.title('Comparison of MAE with increasing batch size', fontsize=16)
plt.xlabel('Batch size', fontsize=14)
plt.ylabel('MAE', fontsize=14)

# # Save the plot
plt.savefig('report/figures/batch_size_vs_mae_plot.png')
