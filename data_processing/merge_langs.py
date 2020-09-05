import pandas as pd
# import sys
import logging
# import os
# sys.path.insert(1, os.getcwd())
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

    esp_df['PennTree POS'] = '_'
    eng_df['Domain'] = 'eng_conll09'
    eng_df = eng_df.reindex([*[eng_df.columns[-1]], *eng_df.columns[:-1]], axis=1)
    esp_df['Domain'] = 'esp_conll09'
    esp_df = esp_df.reindex([*[esp_df.columns[-1]], *esp_df.columns[:-1]], axis=1)

    merge_and_save(eng_df, esp_df, os.path.join(os.getcwd(), '..', 'Data', 'Final') , f'eng09-esp17_{split}.txt')

def merge_09_09(split):
    eng_df = load_normalized(split, 'English')
    esp_df = load_normalized(split, 'Spanish')

    eng_df['Domain'] = 'eng_conll09'
    eng_df = eng_df.reindex([*[eng_df.columns[-1]], *eng_df.columns[:-1]], axis=1)
    esp_df['Domain'] = 'esp_conll09'
    esp_df = esp_df.reindex([*[esp_df.columns[-1]], *esp_df.columns[:-1]], axis=1)

    filepath = os.path.join(os.getcwd(), '..', 'Data', 'Final')
    merge_and_save(eng_df, esp_df, filepath, f'eng09-esp09_{split}.txt')

if __name__ == '__main__':
    merge_09_09('test')
    merge_09_09('dev')
    merge_09_09('train')
    merge_09_17('train')
    merge_09_17('dev')
    merge_09_17('test')

    # TODO: Esp word ids start from 1, Eng starts from 0. Check if this is a problem, and change this.
    print('Done.')


    # TODO: Change this file to ensure it fits in with the new Spanish format.