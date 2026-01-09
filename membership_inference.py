import pandas as pd  
import numpy as np

class All_data():
    def __init__(self, original_data, synthetic, alternative_data):
        self.original = original_data
        self.alt_data = alternative_data
        self.synthetic = synthetic


def count_matches(original: pd.DataFrame, synth: pd.DataFrame, dec: int = 2):
    '''
    For every variable count whether their value occurs in the synthetic data.
    :param original: pd.DataFrame with original data
    :param synth: pd.DataFrame with synthetic or alternative data
    :param dec: Number of decimals on which we match values, defaults to 2.
    :return: Numpy array with number of mathces per individual in the original data.
    '''
    
    n, d = original.shape
    matches_per_var = np.zeros((n, d))
    
    for i, col in enumerate(original.columns):
        if col in synth.columns:    
            if original[col].dtype == 'float64' and synth[col].dtype == 'float64':
                print(col)
                matches_per_var.T[i] = np.isin(np.round(original[col], dec), np.round(synth[col], dec))
            else:
                matches_per_var.T[i] = np.isin(original[col].astype(str), synth[col].astype(str))
        if np.sum(matches_per_var.T[i]) == 0:
            print(f"No matches found for variable {col}, this could be due to different formatting.")
    return np.sum(matches_per_var, 1)

which = lambda lst:list(np.where(lst)[0])

def mi_predictions(matches):
    '''
    Based on an array of number of matches, assign each value to original
    or alternative, such that 50% has been correctly assigned.
    :param matches: numpy array or pd.Series with the number of matches for both
        original and alternative individuals.
    :return: The predictions for each of the values. 1=original data, 0=alternative data.

    '''
    predictions = np.zeros(len(matches))
    larger_than = np.zeros(10)
    for i in range(10):
        larger_than[i] = np.mean(matches>=i)
    diff = larger_than-0.5
    
    if all(diff>=0):
        prop1 = 0.5/np.mean(matches == 9)
        prop0 = 1 - prop1
        predictions[matches == 9] = np.random.choice([0,1], p=[prop0, prop1], 
                                                                 size=np.sum(matches==9))
    else:
        val = max([x for x in diff if x < 0])
        threshold = which(diff == val)[0]
        prop_missing = 0.5-larger_than[threshold]
        prop_below_threshold = np.mean(matches==(threshold-1))
        prop = prop_missing/prop_below_threshold

        predictions[matches >= threshold] = 1
        predictions[matches == (threshold-1)] = np.random.choice([0,1], p=[1-prop, prop], 
                                                                 size=np.sum(matches==(threshold-1)))

    return predictions




def membership_inference(filepath_synthetic: str, 
                         filepath_original: str, 
                         filepath_alternative: str,
                         filepath_results: str,
                         dec: int=3, 
                         n: int=0):
    '''
    Compute success rate of membership inference attack. For every person in the
    original and alternative data set, we compute how many of their values occur
    in the synthetic data (when rounding to 'dec' decimals). The number of variables
    on which a match can be found is their membership inference (mi) score.
    We use the percentage correct to compute how accurately these scores enable an attacker
    to distinguish between individuals from the alternative and original data.
    With percentage at risk we indicate the proportion of individuals that
    an attacker can correctly identify as being part of the original data.
    
    :param filepath_synthetic: path to csv with synthetic data
    :param filepath_original: path to csv with original data    
    :param filepath_alternative: path to csv with alternative data. Ideally this is another sample
    from the population. However, it can also be a holdoutset. 
    :param filepath_results: path to folder where results will be saved.
    :param dec: Number of decimals on which we match values, defaults to 2.
    :param n: Sample size of alternative and original data that will be checked, defaults to 0, in which
    case the full data sets are used.
    :return: Two pandas dataframes. One with pc score per synthetic data set
        and one with membership inference scores per synthetic data set.

    NOTE: it might be required to change the separator in the pd.read_csv commands.
    NOTE: it's important that the formatting of the date variables is the same in all three data sets.

    '''
    original_data = pd.read_csv(filepath_original, sep=';', encoding='utf-8')
    synthetic_data = pd.read_csv(filepath_synthetic, encoding='utf-8')
    alternative_data = pd.read_csv(filepath_alternative, encoding='utf-8')

    all_data = All_data(original_data, synthetic_data, alternative_data)

    mi_score = pd.DataFrame()
    perc_correct =  pd.DataFrame()
    perc_at_risk = pd.DataFrame()
    if n == 0:
        n, d = all_data.original.shape

    for attr, value in all_data.__dict__.items():
        if str(attr) != 'original':
            mi_score[str(attr)] = count_matches(all_data.original,
                                value[0:n],
                                dec=dec)

    # Then we set a threshold such that the alternative and synth records
    # are distributed as close to 50-50 as possible.  
    max_alt = mi_score.alt_data.max()          
    true_labels = np.hstack([np.ones(n), np.zeros(n)])
    all_matches = mi_score[['alt_data', 'synthetic']].to_numpy().ravel()  # or .reshape(-1)
    predictions = mi_predictions(all_matches)
    perc_correct = np.mean(predictions[0:n])
    perc_at_risk = np.mean(mi_score['synthetic']>max_alt)
    
    result = pd.DataFrame({'perc_correct': [perc_correct],
                  'perc_at_risk': [perc_at_risk]})
    result.to_csv(filepath_results + 'membership_inference.csv', index=False, na_rep='NA')

    return result

if __name__ == "__main__":
    '''
    # Example usage (fill in paths and uncomment section):

    filepath_orig = **Please fill in string to location of original data as provided by Lotte**
    filepath_synth = **Please fill in string to location of csv with synthetic data**
    filepath_alt = **Please fill in string to location of csv with alternative (holdout) data**

    print(membership_inference(filepath_synth, filepath_orig, filepath_alt, filepath_results, dec=2, n=1000))
    '''
