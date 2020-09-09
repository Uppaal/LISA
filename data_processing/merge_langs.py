import pandas as pd
import logging
import os
from data_processing.load_utils import *
logging.basicConfig(level=logging.INFO)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def merge_and_save(eng_df, esp_df, filepath, filename):
    final_df = eng_df.append(esp_df)
    if len(eng_df.columns) >= len(esp_df.columns):
        final_df = final_df.reindex(eng_df.columns, axis=1)
    else:
        final_df = final_df.reindex(esp_df.columns, axis=1)
    final_df.to_csv(os.path.join(filepath, filename), sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
    add_blank_line_after_sent(filename)


def merge_09_17(split):
    eng_df = load_normalized(split, 'English')
    esp_df = load_17_spanish(split)

    filepath = os.path.join(DATADIR, 'Final')
    merge_and_save(eng_df, esp_df, filepath , f'eng09-esp17_{split}.txt')

def merge_09_09(split):
    eng_df = load_normalized(split, 'English')
    esp_df = load_normalized(split, 'Spanish')

    filepath = os.path.join(DATADIR, 'Final')
    merge_and_save(eng_df, esp_df, filepath, f'eng09-esp09_{split}.txt')

if __name__ == '__main__':
    # TODO: Esp word ids start from 1, Eng starts from 0. Check if this is a problem, and change this.
    # merge_09_09('test')
    # merge_09_09('dev')
    # merge_09_09('train')
    # merge_09_17('train')
    # merge_09_17('dev')
    # merge_09_17('test')
    print('Done.')
