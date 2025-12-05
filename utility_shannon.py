import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def salary_per_instelling(df):
    return (df['BRUTSAL'] * df['OMVBTR']).groupby(df['INSTELLING']).mean()

def transform_dates(df):
    for varname in ['DATEIND', 'DATBEG', 'gebjaarmaand']:
        if varname == 'gebjaarmaand':
            # Special handling for gebjaarmaand in format YYYYMM
            df[varname] = pd.to_datetime(df[varname].astype(str), format='%Y%m', errors='coerce')
        else:   
            df[varname] = pd.to_datetime(df[varname], format='%Y%m%d')
    return df

def data_specific_utility_df(df, orig, filepath_results, names=None, save=True):
    """
    Calculate the mean absolute difference in average salary per institution between the original 
    and synthetic dataset(s). The outcome is saved in the indicated folder as a CSV file.
    
    Parameters:
    - df: DataFrame or list of DataFrames
    - orig: Original DataFrame
    - names: Optional list of names (same length as df if df is a list), e.g. ["method1", "method2"]
    
    Returns:
    - If df is a single DataFrame: float (mean difference)
    - If df is a list: DataFrame with names and differences
    """
   

    def compute_difference(synth_df):
        orig_means  = salary_per_instelling(orig)
        synth_means = salary_per_instelling(synth_df)
        # Align institutions safely
        aligned = orig_means.align(synth_means, join='outer', fill_value=np.nan)
        comparison = (aligned[0] - aligned[1]).abs()
        return comparison.mean(skipna=True)

    
    # Case 1: Single DataFrame
    if isinstance(df, pd.DataFrame):
        return compute_difference(df)
    
    # Case 2: List of DataFrames
    elif isinstance(df, list):
        if names and len(names) != len(df):
            raise ValueError("Length of names must match length of df list.")
        
        results = []
        for i, synth_df in enumerate(df):
            diff = compute_difference(synth_df)
            label = names[i] if names else f"Dataset_{i+1}"
            results.append({'name': label, 'difference': diff})
        if save:
            pd.DataFrame(results).to_csv(filepath_results + 'avg_sal_per_institution.csv', index=False, na_rep='NA')

        return pd.DataFrame(results)
    
    else:
        raise TypeError("df must be a DataFrame or a list of DataFrames.")


def data_specific_utility(filepath_synthetic, filepath_original, filepath_results, sep_synth=',', names=None, save=True):
    """
    Calculate the mean absolute difference in average salary per institution between the original 
    and synthetic dataset(s). The outcome is saved in the indicated folder as a CSV file.
    
    Parameters:
    - df: DataFrame or list of DataFrames
    - orig: Original DataFrame
    - names: Optional list of names (same length as df if df is a list), e.g. ["method1", "method2"]
    
    Returns:
    - If df is a single DataFrame: float (mean difference)
    - If df is a list: DataFrame with names and differences
    """
    orig = pd.read_csv(filepath_original, sep=';', encoding='utf-8', low_memory=False)
    orig['BRUTSAL'] =  orig['BRUTSAL'].str.replace(',', '.').astype('float64')
    orig['OMVBTR'] =  orig['OMVBTR'].str.replace(',', '.').astype('float64')
    orig['OMVDIO'] =  orig['OMVDIO'].str.replace(',', '.').astype('float64')

    synth_df = pd.read_csv(filepath_synthetic, sep=sep_synth, encoding='utf-8', low_memory=False)
            
    def compute_difference(synth_df):
        orig_means  = salary_per_instelling(orig)
        synth_means = salary_per_instelling(synth_df)
        # Align institutions safely
        aligned = orig_means.align(synth_means, join='outer', fill_value=np.nan)
        comparison = (aligned[0] - aligned[1]).abs()
        return comparison.mean(skipna=True)

    # Case 1: Single DataFrame
    return compute_difference(synth_df)
    

def plot_spec_variable_pairs(filepath, filepath_results, sep_synth = ',', name = "original_data", extract_date=False):
    """
    This plot generates several plots for specific variable pairs in the data.
    Parameters:
    - filepath: path to the CSV file containing the data.
    - filepath_results: path to the folder where the plots will be saved.
    - sep_synth: separator used in the synthetic data CSV file.
    - name: A string indicating the name of the dataset. This
    name should contain the method and parameter values used when it is not original data.
    
    """

    
    if name == "original_data":
        df = pd.read_csv(filepath, sep=';', encoding='utf-8', low_memory=False)
        df['BRUTSAL'] = df['BRUTSAL'].str.replace(',', '.').astype('float64')
        df['OMVBTR'] = df['OMVBTR'].str.replace(',', '.').astype('float64')
        df['OMVDIO'] = df['OMVDIO'].str.replace(',', '.').astype('float64')
        df['DATEIND'] = pd.to_datetime(df['DATEIND'], format='%Y%m%d', errors='coerce')
    else:
        df = pd.read_csv(filepath, sep=sep_synth, encoding='utf-8', low_memory=False)
    print("start dateind")
    print(df['DATEIND'])


    # Extract only the date part
    if extract_date:
        df['DATEIND'] = pd.to_datetime(df['DATEIND'], errors='coerce')
        df['DATEIND'] = df['DATEIND'].dt.date
    print(df['DATEIND'])

    sns.violinplot(x=df['FCAT'], y=df['DATEIND'])
    plt.title("Violin: FCAT and DATEIND for " + name)
    plt.savefig(filepath_results + "DATEIND_FCAT" + name + ".png")
    plt.close()

    order = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16','17', '18', 'BCAO', 'HO11', 'HO12', 'LB', 'LC', 'LD', 'LE', 'LIO', 'ML', 'MLBB', 'P', 'XXX']    
    print("start SALSCH")

    plt.figure(figsize=(14, 6))  # Wider plot
    sns.violinplot(x='SALSCH', y='BRUTSAL', data=df, order=order)
    plt.title("Violin: Salary Scale and Bruto Salary for " + name)
    plt.xticks(rotation=45)  # Rotate labels for readability
    plt.tight_layout()
    plt.savefig(filepath_results + f"SALSCH_BRUTSAL" + name + ".png")
    plt.close()
    print("start OMVDIO")

    sns.scatterplot(x=df['OMVDIO'], y=df['OMVBTR'], alpha=0.05)
    plt.title("Scatterplot: OMVDIO and OMVBTR for " + name)
    plt.savefig(filepath_results + f"OMVDIO_OMVBTR" + name + ".png")
    plt.close()
    print("start DATBEG")

    sns.scatterplot(x=df['DATBEG'], y=df['OMVBTR'], alpha=0.05)
    plt.title("Scatterplot: DATBEG and OMVBTR for " + name)
    plt.savefig(filepath_results + f"DATBEG_OMVBTR" + name + ".png")
    plt.close()

if __name__ == "__main__":

    filepath_results = "C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/results/"
    filepath_orig = "C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/data/synthetische_data_MBO_personeel_2025/synthetische_data_MBO_personeel_2025_pd202410.csv"
    filepath_synth = "C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/data/synthetic_postprocessed/synth_t_65.csv"
    plot_spec_variable_pairs(filepath_orig, filepath_results, name="original_data")    
    # plot_spec_variable_pairs(filepath_synth, filepath_results, name="synthetic t 65")    

    # data_specific_utility(filepath_synth, filepath_orig, filepath_results, names=["synth_t_65"], save=True)
    plot_spec_variable_pairs(filepath_synth, filepath_results, name="synthetic t 65", extract_date=True)    

    # data_specific_utility(filepath_synth, filepath_orig, filepath_results, names=["synth_t_65"], save=True)
