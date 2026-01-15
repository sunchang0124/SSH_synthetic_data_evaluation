import pandas as pd
import matplotlib.pyplot as plt
import os
file_path = "C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/realdata/results"
synth = pd.read_csv(file_path + "synth.csv", sep=',', encoding='utf-8')
orig = pd.read_csv(file_path + "prepped_data.csv", sep=',', encoding='utf-8')
synth65 = pd.read_csv(file_path + "synth_t_65.csv", sep=',', encoding='utf-8')
synth25 = pd.read_csv(file_path + "synth_t_25.csv", sep=',', encoding='utf-8')
# synth = pd.read_csv(file_path + "synth.csv", sep=',', encoding='utf-8')
synth1 = pd.read_csv(file_path + "synth1cluster.csv", sep=',', encoding='utf-8')


file_path_orig = "C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/data/synthetische_data_MBO_personeel_2025/synthetische_data_MBO_personeel_2025_pd202410.csv"
raw_orig = pd.read_csv(file_path_orig, sep=';', encoding='utf-8')
raw_orig['BRUTSAL'] = raw_orig['BRUTSAL'].str.replace(',', '.').astype('float64')
raw_orig['OMVBTR'] = raw_orig['OMVBTR'].str.replace(',', '.').astype('float64')
raw_orig['OMVDIO'] = raw_orig['OMVDIO'].str.replace(',', '.').astype('float64')

  

variable_schema = {
    'INSTELLING': 'categorical',   # Likely institution code/name
    'BESTUUR': 'categorical',      # Governing body or board
    'SALSCH': 'categorical',       # Salary scale (categorical)
    'SALNUM': 'numeric',           # Salary number (numeric)
    'FSCHAL': 'categorical',       # Possibly a classification scale
    'AARD': 'categorical',         # Nature/type of contract
    'BRUTSAL': 'numeric',          # Gross salary (numeric)
    'GESLACHT': 'categorical',     # Gender (categorical)
    'FCAT': 'categorical',         # Function category
    'OMVBTR': 'numeric',       # Possibly organizational attribute
    'OMVDIO': 'numeric',       # Another organizational attribute
    'gebjaarmaandDSE': 'numeric',     # Birth year-month (numeric or date)
    'DATEINDDSE': 'numeric',      # End date (better as date, but miceforest treats as categorical)
    'DATBEGDSE': 'numeric'        # Start date (same as above)
}
def plot_comparison(orig, synth, name=""):
    # Assume both dataframes have the same columns
    for col in orig.columns:
        print(variable_schema[col])
        print(pd.api.types.is_numeric_dtype(orig[col]))
        # Check if the column is numeric
        if variable_schema[col] == 'numeric':
            # Plot overlapping histograms
            plt.hist(orig[col].dropna(), bins=15, alpha=0.5, label='orig', color='blue')
            plt.hist(synth[col].dropna(), bins=15, alpha=0.5, label='synth', color='red')
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

        else:
            # Plot overlapping bar plots for categorical data
            orig_counts = orig[col].value_counts()
            synth_counts = synth[col].value_counts()
            categories = list(set(orig_counts.index).union(set(synth_counts.index)))
            
            orig_vals = [orig_counts.get(cat, 0) for cat in categories]
            synth_vals = [synth_counts.get(cat, 0) for cat in categories]
            
            x = range(len(categories))
            
            plt.bar(x, orig_vals, alpha=0.5, label='orig', color='blue')
            plt.bar(x, synth_vals, alpha=0.5, label='synth', color='red')
            plt.xticks(ticks=x, labels=categories, rotation=45)
            plt.title(f'Overlapping Bar Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
        plt.legend()
        plt.savefig(file_path + "figures/"+ f"{col}" + name + ".png")
        plt.close()


# plot_comparison(orig, synth)
# plot_comparison(orig, synth65, name="_t65")
# plot_comparison(orig, synth1, name="_t1")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_pairs(df, name):
    """
    Generate plots for every pair of variables in the dataframe.
    - Skip variables with more than 20 unique categories.
    - For continuous-continuous pairs: scatterplot.
    - For discrete-continuous pairs: violin plot.
    - For discrete-discrete pairs: heatmap of occurrences.
    """
    # Helper function to determine if a variable is discrete
    def is_discrete(series):
        return series.dtype == 'object' or pd.api.types.is_categorical_dtype(series) or series.nunique() <= 20

    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            x = df[columns[i]]
            y = df[columns[j]]

            # Skip if either variable has too many categories
            if is_discrete(x) and x.nunique() > 20:
                continue
            if is_discrete(y) and y.nunique() > 20:
                continue

            plt.figure(figsize=(8, 6))
            if not is_discrete(x) and not is_discrete(y):
                sns.scatterplot(x=x, y=y)
                plt.title(f"Scatterplot: {columns[i]} vs {columns[j]}")
            elif is_discrete(x) and not is_discrete(y):
                sns.violinplot(x=x, y=y)
                plt.title(f"Violin Plot: {columns[j]} by {columns[i]}")
            elif not is_discrete(x) and is_discrete(y):
                sns.violinplot(x=y, y=x)
                plt.title(f"Violin Plot: {columns[i]} by {columns[j]}")
            else:
                # Both are discrete
                cross_tab = pd.crosstab(x, y)
                sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Heatmap: {columns[i]} vs {columns[j]}")

            plt.tight_layout()
            filename = file_path + "figures/"+ str(x.name) + str(y.name) + name + ".png"
            print(filename)
            plt.savefig(filename)

            # plt.show()

# plot_variable_pairs(synth25, name="synth25")
# plot_variable_pairs(orig, name = "orig")
# plot_variable_pairs(synth65, name="synth65")

def salary_per_instelling(df):
    df['BRUTSAL*OMVBTR'] = df['BRUTSAL'] * df['OMVBTR']
    mean_values = df.groupby('INSTELLING')['BRUTSAL*OMVBTR'].mean()
    return mean_values


def data_specfic_utility(df):
    '''
    Calculate the mean absolute difference in average salary per institution between the original and synthetic datasets.
    '''
    orig_means = salary_per_instelling(orig)
    synth_means = salary_per_instelling(df)
    comparison = pd.DataFrame({'orig_mean': orig_means, 'synth_mean': synth_means})
    comparison['difference'] = abs(comparison['orig_mean'] - comparison['synth_mean'])

    # print(comparison['difference'].mean())
    return comparison['difference'].mean()


data_specfic_utility(orig)
data_specfic_utility(synth25)
data_specfic_utility(synth65)
data_specfic_utility(synth1)

# Specific two-dimensional plots


def plot_spec_variable_pairs(df, name):
    """
    name should containt the method and parametervalues used
    """
    # sns.scatterplot(x=df['DATEIND'], y=df['FCAT'], alpha=0.05)
    sns.violinplot(x=df['FCAT'], y=df['DATEIND'])
    plt.title("Violin: FCAT and DATEIND")
    plt.savefig(file_path + "figures/"+ f"DATEIND_FCAT" + name + ".png")
    plt.close()

    order = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16','17', '18', 'BCAO', 'HO11', 'HO12', 'LB', 'LC', 'LD', 'LE', 'LIO', 'ML', 'MLBB', 'P', 'XXX']    
    
    plt.figure(figsize=(14, 6))  # Wider plot
    sns.violinplot(x='SALSCH', y='BRUTSAL', data=df, order=order)
    plt.title("Violin: Salary Scale and Bruto Salary")
    plt.xticks(rotation=45)  # Rotate labels for readability
    plt.tight_layout()
    plt.savefig(file_path + "figures/"+ f"SALSCH_BRUTSAL" + name + ".png")
    plt.close()

    order = df['SALSCH'].unique().tolist()

    sns.scatterplot(x=df['OMVDIO'], y=df['OMVBTR'], alpha=0.05)
    plt.title("Scatterplot: OMVDIO and OMVBTR")
    plt.savefig(file_path + "figures/"+ f"OMVDIO_OMVBTR" + name + ".png")
    plt.close()

    sns.scatterplot(x=df['DATBEG'], y=df['OMVBTR'], alpha=0.05)
    plt.title("Scatterplot: DATBEG and OMVBTR")
    plt.savefig(file_path + "figures/"+ f"DATBEG_OMVBTR" + name + ".png")
    plt.close()

plot_spec_variable_pairs(raw_orig, name="raw_orig")    