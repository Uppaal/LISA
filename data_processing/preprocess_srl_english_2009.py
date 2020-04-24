import pandas as pd
import sys
import logging
from Scratch.load_utils import *
from Scratch.sentence_manipulation import Tree_Manipulation
import numpy as np
logging.basicConfig(level=logging.INFO)
sys.path.insert(1, os.getcwd())
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

srl_cols = ['SRL-1', 'SRL-2', 'SRL-3', 'SRL-4', 'SRL-5', 'SRL-6', 'SRL-7', 'SRL-8', 'SRL-9', 'SRL-10', 'SRL-11',
                'SRL-12', 'SRL-13', 'SRL-14', 'SRL-15', 'SRL-16', 'SRL-17', 'SRL-18', 'SRL-19', 'SRL-20', 'SRL-21',
                'SRL-22', 'SRL-23', 'SRL-24', 'SRL-25', 'SRL-26']
universal_srl_tagset = ['A0','A1','A2','A3','A4','A5']

def normalize_srl_tagset(df):
    def normalize_english(x):
        if x is None or pd.isnull(x):
            return None
        if x == 'O' or x == 'B-V':
            return x
        if '-' in x:
            if x.split('-')[1] not in universal_srl_tagset:
                return 'O'
            return str(x.split('-')[0] + '-' + x.split('-')[1])
        return None

    num_srl_cols = len(df.columns[list(df.columns).index('SRL-1'):])
    for i in range(num_srl_cols):
        try:
            df[srl_cols[i]] = df[srl_cols[i]].apply(normalize_english).values
        except Exception as e:
            print(f"Error in Normalizing SRL tags for column number {i}, {srl_cols[i]}. \n{e}")

    print(f"Tagset reduced to: {sorted(list(pd.unique(df[srl_cols[:num_srl_cols]].values.ravel())), key=lambda x: (x is None, x))}")
    return df


def copy_srl_tags(df_05, df):
    max_sent_id = max(set(df['Sent ID']))
    num_srl_cols = len(df_05.columns[list(df_05.columns).index('SRL-1'):])
    target_columns = ['Sent ID', 'ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense']+srl_cols[:num_srl_cols]
    df_target = pd.DataFrame(columns=target_columns)

    def predicate_sense(temp):
        if temp['Is Predicate'] == '_' and temp['Predicate'] != '-':
            logging.warning('05 has predicates not present in 09.')
            assert False
        if temp['Predicate'] == '-':
                return '_'
        return temp['Predicate.Verb Sense']

    def predicate(temp):
        if temp['Predicate.Verb Sense'] == '_':
            return '_'
        else:
            return 'Y'

    _05_sent_id = 0
    for sent_id in range(1, max_sent_id):

        while list(df[df['Sent ID'] == sent_id]['Form'].values) != list(df_05[df_05['Sent ID']==_05_sent_id]['Form'].values):
            logging.info(f"Adding 05 sent id by 1, from {_05_sent_id} to {_05_sent_id+1}")
            _05_sent_id += 1

        logging.info(f'Found sent match at {_05_sent_id}')
        df_sent = df[df['Sent ID'] == sent_id][target_columns[0:9]]

        # Copy Predicates
        temp = pd.DataFrame({'Predicate': list(df_05[df_05['Sent ID']==_05_sent_id]['Predicate'].values),
                             'Is Predicate': list(df_09[df_09['Sent ID']==sent_id]['Is Predicate'].values),
                             'Predicate.Verb Sense': list(df_09[df_09['Sent ID']==sent_id]['Predicate.Verb Sense'].values)})
        df_sent['Predicate.Verb Sense'] = temp.apply(predicate_sense, axis=1).values
        df_sent['Is Predicate'] = df_sent.apply(predicate, axis=1).values

        srl_sent_count = 0
        while set(pd.notna(df_05[df_05['Sent ID']==_05_sent_id][srl_cols[srl_sent_count]])) != {False}:
            srl_sent_count+=1
        assert np.count_nonzero(df_sent['Is Predicate'].values == 'Y') == srl_sent_count

        # Copy SRL
        for j in range(srl_sent_count):
            df_sent[srl_cols[j]] = df_05[df_05['Sent ID']==_05_sent_id][srl_cols[j]].values
            assert np.count_nonzero(df_sent[srl_cols[j]].values == 'B-V') == 1

        df_target = df_target.append(df_sent)
        df_target = df_target[target_columns]
        _05_sent_id += 1

    return df_target


if __name__ == "__main__":
    split = 'train'
    df_05 = load_05(split)
    df_09 = load_09_english_ud(split)
    df_target = copy_srl_tags(df_05, df_09)
    df_target.to_csv(os.path.join(os.getcwd(), '..', 'Data', 'English', '2009_normalized', f'converted_{split}.txt'),
                     sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
    df_target = normalize_srl_tagset(df_target)
    df_target.to_csv(os.path.join(os.getcwd(), '..', 'Data', 'English', '2009_normalized', f'converted_{split}.txt'), sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
    logging.info("Done.")
