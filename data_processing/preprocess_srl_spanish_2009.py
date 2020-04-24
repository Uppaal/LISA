import pandas as pd
import sys
import logging
from Scratch.load_utils import *
from Scratch.sentence_manipulation import Tree_Manipulation
logging.basicConfig(level=logging.INFO)
sys.path.insert(1, os.getcwd())
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


srl_cols = ['SRL-1', 'SRL-2', 'SRL-3', 'SRL-4', 'SRL-5', 'SRL-6', 'SRL-7', 'SRL-8', 'SRL-9', 'SRL-10', 'SRL-11',
                'SRL-12', 'SRL-13', 'SRL-14', 'SRL-15', 'SRL-16', 'SRL-17', 'SRL-18', 'SRL-19', 'SRL-20', 'SRL-21',
                'SRL-22', 'SRL-23', 'SRL-24', 'SRL-25', 'SRL-26']
spanish_to_universal_tag = {'arg0': 'A0', 'arg1': 'A1', 'arg2': 'A2', 'arg3': 'A3', 'arg4': 'A4', 'argM': 'O', 'argL': 'O'}


def normalize_srl_tagset(df):
    def normalize_spanish(x):
        if x is None or pd.isnull(x):
            return None
        if x == '_':
            return 'O'
        if '-' in x:
            return spanish_to_universal_tag[x.split('-')[0]]
        return None

    num_srl_cols = len(df.columns[list(df.columns).index('SRL-1'):])
    for i in range(num_srl_cols):
        try:
            df[srl_cols[i]] = df[srl_cols[i]].apply(normalize_spanish)
        except:
            print(f"Error in Normalizing SRL tags for row number {i}, {srl_cols[i]}")

    print(f"Tagset reduced to: {list(pd.unique(df[srl_cols[:num_srl_cols]].values.ravel()))}")
    return df


def convert(df):
    def convert_srl(sent_id):
        t = Tree_Manipulation(sent_id=sent_id, df=df)
        root = t.build_tree_for_sent()
        return t.change_srl_to_bio()

    bad_srl = 0
    df_target = pd.DataFrame(columns=df.columns)
    max_sent_id = max(set(df['Sent ID']))
    for i in range(1, max_sent_id+1):
        try:
            df_sent = df[df['Sent ID'] == i][df.columns[0:9]]
            srl_tags = convert_srl(i)

            # Check for issues
            if not all([True if 'B-V' in l else False for l in srl_tags]):
                bad_srl += 1
                logging.warning(f"'B-V' was overwritten in sentence {i}. Skipping.")
                continue

            # Add to final dataframe
            for j in range(len(srl_tags)):
                df_sent[srl_cols[j]] = srl_tags[j]
            df_target = df_target.append(df_sent)
            df_target = df_target[df.columns]

        except Exception as e:
            logging.warning(f"ERROR in sentence {i}:")
            logging.warning(e)
            continue

    print(f"'B-V' tag was overwritten in {bad_srl}/{max_sent_id} sentences.")
    return df_target

def check(sent_id):
    df_sent = df[df['Sent ID'] == sent_id]
    t = Tree_Manipulation(sent_id=sent_id, df=df)
    root = t.build_tree_for_sent()
    tags = t.change_srl_to_bio()
    logging.info(f"{not all([True if 'V' in l else False for l in tags])} should be False")


if __name__ == "__main__":
    df = load_09_spanish('test')
    df_target = normalize_srl_tagset(df)
    df_target = convert(df_target)
    df_target.to_csv(os.path.join(os.getcwd(), '..', 'Data', 'Spanish', '2009_normalized', 'converted_test.txt'), sep='\t', header=False, index=False, encoding='utf-8', quoting=csv.QUOTE_NONE)
    print("Done.")
