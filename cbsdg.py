# -*- coding: utf-8 -*-
"""
These files contain functions to generate synthetic data with CBSDG.
"""
import copy
import pickle
import time
import numpy as np
import pandas as pd
import miceforest as mf
from spn.structure.StatisticalTypes import MetaType
from spn.structure.Base import Context


def compute_days_since(df, varnames):
    # varname can be multiple variables

    for varname in varnames:
        # Convert to 
        if varname == 'gebjaarmaand':
            # Special handling for gebjaarmaand in format YYYYMM
            df[varname] = pd.to_datetime(df[varname].astype(str), format='%Y%m', errors='coerce')

        else:   
            df[varname] = pd.to_datetime(df[varname], format='%Y%m%d')
        # Compute days since earliest date
        earliest_date = df[varname].min()
        # DSE stands for days since earliest.
        df[varname + 'DSE'] = (df[varname] - earliest_date).dt.days
    return df.drop(columns=varnames)

def reverse_days_since(df, df_orig, varnames):
    """
    Reconstruct original date columns from DSE columns.
    Assumes DSE columns were created using compute_days_since().
    df is the synthetic data and df_orig is the original data.
    """
    for varname in varnames:
        dse_col = varname + 'DSE'
        if dse_col not in df.columns:
            continue  # Skip if DSE column is missing

        # Convert to 
        if varname == 'gebjaarmaand':
            # Special handling for gebjaarmaand in format YYYYMM
            # added .dt.to_period('M') : need to check if it works:
            df_orig[varname] = pd.to_datetime(df_orig[varname].astype(str), format='%Y%m', errors='coerce')
        else:   
            df_orig[varname] = pd.to_datetime(df_orig[varname], format='%Y%m%d')

            earliest_date = df_orig[varname].min()
        # If you want the entire row:
        # earliest_date = df_orig[varname].loc[earliest_index]

        # Reconstruct the original date

        df[varname] = earliest_date + pd.to_timedelta(df[dse_col].astype('float'), unit='D')
        df[varname] = df[varname].dt.normalize()

    return df

def reverse_category_encoding(df, cat_cols, mappings):
    """
    df: DataFrame with encoded columns
    cat_cols: list of categorical column names
    mappings: dict with {column_name: list_of_original_categories}
    """
    for col in cat_cols:
        if col in mappings:
            # Convert codes back to categories using stored mapping
            df[col] = df[col].astype('Int64')  # Pandas nullable integer type
            df[col] = pd.Categorical.from_codes(df[col], categories=mappings[col])
    return df
def postprocess_existing_synth(data, first_path = r"C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/data/synthetic_processed/", 
                               second_path = r"C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/data/synthetic_postprocessed/"):
    
    prepped_data, col_mapping = preprocessing(data)

    for filename in os.listdir(first_path):
        full_path = os.path.join(first_path, filename)
        if os.path.isfile(full_path):  # Optional: skip subdirectories
            synth = pd.read_csv(first_path + filename, sep=',', encoding='utf-8')
            synth_postprocessed = postprocessing(data, col_mapping, synth)
            synth_postprocessed.to_csv(second_path + filename, index=False, na_rep='NA')



def preprocessing(df):
    # Remove peilmaand
    df_raw = df.drop(columns=['PEILMND', 'CDSADM', 'LEVERAN'])
    df = compute_days_since(df_raw, ['DATEIND', 'DATBEG', 'gebjaarmaand'])
    # Encode categorical variables
    mappings = {}
    cat_cols =['INSTELLING', 'BESTUUR', 'SALSCH', 'SALNUM', 'FSCHAL', 'AARD', 'GESLACHT', 'FCAT']
    for col in cat_cols:
        df[col] = df[col].astype('category')
        mappings[col] = df[col].cat.categories.tolist()  # Save original categories
        df[col] = df[col].cat.codes

    # Change decimal indicator.
    df['BRUTSAL'] = df['BRUTSAL'].str.replace(',', '.').astype('float64')
    df['OMVBTR'] = df['OMVBTR'].str.replace(',', '.').astype('float64')
    df['OMVDIO'] = df['OMVDIO'].str.replace(',', '.').astype('float64')

    # Impute missing values using MICE
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
        'OMVDIO': 'categorical',       # Another organizational attribute
        'gebjaarmaandDSE': 'numeric',     # Birth year-month (numeric or date)
        'DATEINDDSE': 'numeric',      # End date (better as date, but miceforest treats as categorical)
        'DATBEGDSE': 'numeric'        # Start date (same as above)
    }
    # Create an imputation kernel
    kernel = mf.ImputationKernel(
        data=df,
        save_all_iterations_data=True,
        random_state=42,

    )
    # Run the MICE algorithm for 5 iterations
    kernel.mice(5)

    # Retrieve the completed dataset
    completed_df = kernel.complete_data(0)

    return completed_df, mappings


def save_object(obj, name='mspn'):
    with open(name + '.pkl', 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def postprocessing(df_orig, col_mappings, df_synth):
    '''
    Reverse preprocessing steps on synthetic data.
    NOTE this does not reverse imputation of missing values, adding
    the removed variables or changing the decimal indicator back.
    
    '''
    # Reverse category encoding
    cat_cols =['INSTELLING', 'BESTUUR', 'SALSCH', 'SALNUM', 'FSCHAL', 'AARD', 'GESLACHT', 'FCAT']
    df_synth = reverse_category_encoding(df_synth, cat_cols, col_mappings)

    # Reverse days since earliest date
    df_synth = reverse_days_since(df_synth, df_orig, ['DATEIND', 'DATBEG', 'gebjaarmaand'])

    return df_synth

def cbsdg(df_orig, var_types, n_synth: int=None, t: int =25, seed: int=1901):

    from spn.algorithms.Sampling import sample_instances
    from numpy.random.mtrand import RandomState
    from spn.algorithms.LearningWrappers import learn_mspn 

    '''
    Generate a synthetic version of a dataframe (df) with cluster-based synthetic data generation

    :param df: Original dataset in pandas dataframe.
    :param var_types:
    :param t: Average number of individuals per cluster, default is 25.
    :param n_synth: Number of synthetic records, defaults to the number of rows in the original data.
    :param seed: 
    :return: Synthetic dataset.

    '''
    n, d = df_orig.shape
    
   # Check if var_types is provided
    if var_types is None:
        raise ValueError(
            "The 'var_types' argument is mandatory. It must be a NumPy array "
            "with one entry per variable, where each entry is either 'discrete' or 'continuous'."
        )

    # Check if all entries are either 'discrete' or 'continuous'
    if not all(v in ['discrete', 'continuous'] for v in var_types):
        raise ValueError("Each entry in 'var_types' must be either 'discrete' or 'continuous'.")

    # Check if the length matches the number of variables
    if len(var_types) != d:
        raise ValueError(
            f"'var_types' must have {data.shape[1]} elements, one for each variable in the data."
        )

    
    if n_synth is None:
        n_synth = copy.deepcopy(n)

    if t == 1:
        mis = n
    else:
        mis = copy.deepcopy(n)-1

    data_np = df_orig.to_numpy()
    ds_context = convert_var_types(var_types)

    # ds_context is a SPFlow specific variable that indicates the variable type. Experiments have shown that it is best
    # to use either discrete or continuous (https://pmc.ncbi.nlm.nih.gov/articles/PMC9748584/)

    # add the domain information based on the numpy array.
    ds_context.add_domains(data_np)
    
    mspn = learn_mspn(data_np, ds_context=ds_context, min_instances_slice=mis, alpha=0,
                      rows="kmeans", threshold = -1, standardize = True, n_clusters=int(n/t), rand_gen=2206)
    
    save_object(mspn, name='mspn' + str(mis) + '_' + str(t)) 
    synth_cbsdg = sample_instances(mspn, np.array([[np.repeat(np.nan,d)] * n_synth]).reshape(-1, data_np.shape[1]), RandomState(seed))
    synth_cbsdg = pd.DataFrame(synth_cbsdg)
    synth_cbsdg.columns = df_orig.columns

    
    return synth_cbsdg

def convert_var_types(var_types):
    
    if not isinstance(var_types, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    mapping = {
        "discrete": MetaType.DISCRETE,
        "continuous": MetaType.REAL
    }

    try:
        return Context(meta_types=np.array([mapping[v] for v in var_types]))
    except KeyError as e:
        raise ValueError(f"Invalid entry in var_types: {e}. Must be 'discrete' or 'continuous'.")

def cbsdg_pre_post(data, n_synth: int=None, t: int =25, seed: int=1901):

    '''
    CBSDG with preprocessing and postprocessing steps.
    Data is the original dataset.
    '''

    prepped_data, col_mapping = preprocessing(data)
    var_types =np.array(['discrete', 'discrete', 'discrete', 'discrete', 'discrete', 'discrete', 'continuous', 'discrete', 'discrete', 
                'continuous', 'discrete', 'continuous', 'continuous', 'continuous'])
    # create the synthetic data that has not been post-processed yet.
    synth_raw = cbsdg(prepped_data, var_types, t=t, seed=seed)
    synth = postprocessing(data, col_mapping, synth_raw)

    return synth, synth_raw