import pandas as pd
from src.cbsdg import cbsdg_pre_post
import time

if __name__ == '__main__':

    file_path = None
    resultfolder = None
    file_path = "C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/data/synthetische_data_MBO_personeel_2025/synthetische_data_MBO_personeel_2025_pd202410.csv"
    resultfolder = r"C:/Users/kroessks/OneDrive - TNO/Documents/Projects/DUO/TDCC/realdata/"

    data = pd.read_csv(file_path, sep=';', encoding='utf-8')
    
    synthmisn, synth_raw = cbsdg_pre_post(data, t=1)
    synthmisn.to_csv(resultfolder + 'synth1cluster.csv', index=False, na_rep='NA')

    start_time = time.time()
    data = pd.read_csv(file_path, sep=';', encoding='utf-8')

    syntht65, synth_raw = cbsdg_pre_post(data, t=65)
    syntht65.to_csv(resultfolder + 'synth_t_65.csv', index=False, na_rep='NA')
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time spent running code 65: {elapsed_time:.4f} seconds")
