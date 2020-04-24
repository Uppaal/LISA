import os
import pandas as pd
import csv

srl_names = ['SRL-1','SRL-2','SRL-3','SRL-4','SRL-5','SRL-6','SRL-7','SRL-8','SRL-9','SRL-10','SRL-11','SRL-12','SRL-13','SRL-14','SRL-15',
                'SRL-16','SRL-17','SRL-18','SRL-19','SRL-20','SRL-21','SRL-22','SRL-23','SRL-24','SRL-25','SRL-26']

official_colnames = ['ID','FORM','LEMMA', 'PLEMMA', 'POS', 'PPOS', 'FEAT', 'PFEAT', 'HEAD', 'PHEAD', 'DEPREL', 'PDEPREL', 'FILLPRED', 'PRED']
imp_official_colnames = ['ID', 'FORM', 'LEMMA', 'POS', 'HEAD', 'DEPREL', 'FILLPRED', 'PRED']

col_lists = {
    '05 english': (['Domain', 'Sent ID', 'ID', 'Form', 'POS', 'PPOS', 'Parse Head', 'Parse Label', 'Ignore1', 'Predicate Sense', 'Predicate', 'Ignore3', 'Ignore4', 'Fake SRL'],
                   ['Domain', 'Sent ID', 'ID', 'Form', 'Predicate Sense', 'Predicate'],
                   ['Domain', 'Sent ID', 'ID', 'Form', 'Predicate Sense', 'Predicate']),
    '09 english': (official_colnames, imp_official_colnames,
                   ['ID', 'Form', 'Lemma', 'PennTree POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense']),
    '09 english ud': (['ID','Form','Lemma', 'PennTree POS', 'POS', '-', 'Parse Head','Parse Label', 'Is Predicate', 'Predicate.Verb Sense'],
                      ['ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense'],
                      ['ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense']),
    '09 spanish': (official_colnames, imp_official_colnames,
                   ['ID', 'Form', 'Lemma', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense']),
    '17 spanish': (['ID', 'Form', 'Lemma', 'POS', 'PPOS', 'FEAT', 'Parse Head', 'Parse Label', 'Ignore1', 'Ignore2'],
                   ['ID', 'Form', 'Lemma', 'POS', 'Parse Head', 'Parse Label'],
                   ['ID', 'Form', 'Lemma', 'POS', 'Parse Head', 'Parse Label']),
    'English': (['Sent ID', 'ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense'],
                ['Sent ID', 'ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense'],
                ['Sent ID', 'ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense']),
    'Spanish': (['Sent ID', 'ID', 'Form', 'Lemma', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense'],
                ['Sent ID', 'ID', 'Form', 'Lemma', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense'],
                ['Sent ID', 'ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label', 'Is Predicate', 'Predicate.Verb Sense'])
}

metadata = {
    '05 english train': ('train-set.gz.parse.sdeps.combined.bio', 18),
    '05 english dev': ('dev-set.gz.parse.sdeps.combined.bio', 12),
    '05 english test': ('test.wsj.gz.parse.sdeps.combined.bio', 9),
    '05 english test ood': ('test.brown.gz.parse.sdeps.combined.bio', 11),

    '09 english train': ('CoNLL2009-ST-English-train.txt', 26),
    '09 english dev': ('CoNLL2009-ST-English-development.txt', 16),
    '09 english test': ('CoNLL2009-ST-evaluation-English.txt', 16),
    '09 english test ood': ('CoNLL2009-ST-evaluation-English-ood.txt', 18),
    '09 english trial': ('CoNLL2009-ST-English-trial.txt', 12),

    '09 english ud train': ('conll2009-train-universal.txt', 26),
    '09 english ud dev': ('conll2009-dev-universal.txt', 19),
    '09 english ud test': ('conll2009-test-universal.txt', 16),
    '09 english ud test ood': ('conll2009-test-ood-universal.txt', 18),

    '09 spanish train': ('CoNLL2009-ST-Spanish-train.txt', 17),
    '09 spanish dev': ('CoNLL2009-ST-Spanish-development.txt', 13),
    '09 spanish test': ('CoNLL2009-ST-evaluation-Spanish.txt', 15),

    '17 spanish train': ('es-ud-train.conllu.txt', 0),
    '17 spanish dev': ('es-ud-dev.conllu.txt', 0),
    '17 spanish test': ('es-ud-test.conllu.txt', 0),

    'English train': ('converted_train.txt', 18),
    'English dev': ('converted_dev.txt', 12),
    'English test': ('converted_test.txt', 9),

    'Spanish train': ('converted_train.txt', 17),
    'Spanish dev': ('converted_dev.txt', 13),
    'Spanish test': ('converted_test.txt', 15),
}


def count_cols(filename, target=False):
    if target:
        f = open(filename, encoding='utf-8')
    else:
        f = open(filename, 'r')
    x = f.readlines()
    f.close()

    max_tab = 0
    for line in x:
        count = line.count('\t')
        if count > max_tab:
            max_tab = count
    print("Max Number of columns: ", max_tab+1)


def add_sentence_ids(df):
    df.insert(0, 'Sent ID', '')
    curr_sent_id = 1
    df.at[0, 'Sent ID'] = curr_sent_id

    for i in range(1, len(df)):
        if df.iloc[i]['ID'] <= df.iloc[i - 1]['ID']:
            curr_sent_id += 1
        df.at[i, 'Sent ID'] = curr_sent_id
    return df


def load_df(dirpath, dataset, split):
    dataset_key = ' '.join([dataset, split])
    filepath = os.path.join(dirpath, metadata[dataset_key][0])
    num_cols = metadata[dataset_key][1]

    if 'english' in dataset_key:
        df_full = pd.read_csv(filepath, sep='\t', header=None, names=col_lists[dataset][0] + srl_names)
    else:
        df_full = pd.read_csv(filepath, sep='\t', header=None, names=col_lists[dataset][0] + srl_names, encoding='utf-8', quoting=csv.QUOTE_NONE)
    if num_cols < len(srl_names):
        assert set(pd.notna(df_full[srl_names[num_cols]])) == {False}

    df = df_full.filter(col_lists[dataset][1] + srl_names[:num_cols], axis=1)
    if dataset == 'Spanish':
        df['PennTree POS'] = '_'
        df = df.reindex(col_lists[dataset][2] + srl_names[:num_cols], axis=1)
    df.columns = col_lists[dataset][2] + srl_names[:num_cols]
    return df


def load_05(split):
    "split: 'train', 'dev', 'test', 'test ood'"
    dirpath = os.path.join(os.getcwd(), '..', 'Data', 'English', 'conll05')
    df = load_df(dirpath=dirpath, dataset='05 english', split=split)
    return df

def load_09_english_pt(split):
    "split: 'train', 'dev', 'test', 'test ood'"
    dirpath = os.path.join(os.getcwd(), '..', 'Data', 'English', 'conll09')
    df = load_df(dirpath=dirpath, dataset='09 english', split=split)
    df = add_sentence_ids(df)
    return df

def load_09_english_ud(split):
    "split: 'train', 'dev', 'test', 'test ood'"
    dirpath = os.path.join(os.getcwd(), '..', 'Data', 'English', 'conll09_ud')
    df = load_df(dirpath=dirpath, dataset='09 english ud', split=split)
    df = add_sentence_ids(df)
    return df

def load_09_spanish(split):
    "split: 'train', 'dev', 'test'"
    dirpath = os.path.join(os.getcwd(), '..', 'Data', 'Spanish', 'CoNLL2009-ST-Spanish')
    df = load_df(dirpath=dirpath, dataset='09 spanish', split=split)
    df = add_sentence_ids(df)
    return df

def load_17_spanish(split):
    "split: 'train', 'dev', 'test'"
    dirpath = os.path.join(os.getcwd(), '..', 'Data', 'Spanish', 'CoNLL_2017')
    df = load_df(dirpath=dirpath, dataset='17 spanish', split=split)
    df = add_sentence_ids(df)
    return df

def preprocess_17_spanish(filename):
    filepath = os.path.join(os.getcwd(), '..', 'Data', 'Spanish', 'CoNLL_2017', filename)
    content = open(filepath, encoding='utf-8').readlines()
    content = [x.strip() for x in content]
    print("Number of lines: ", len(content))
    # lines = [line for line in content if line!='' and line[0]!='#' and '-' not in line[0]]
    lines=[]
    for line in content:
        if line != '':
            if line[0]!= '#' and '-' not in line.split('\t')[0]:
                lines.append(line)
    print(len(lines), " lines retained.")

    with open(filepath+'.txt', 'w', encoding='utf-8') as f:
        for item in lines:
            f.write("%s\n" % item)
    print('Saved as ', filepath+'.txt')

def load_normalized(split, language):
    dirpath = os.path.join(os.getcwd(), '..', 'Data', language, '2009_normalized')
    df = load_df(dirpath=dirpath, dataset=language, split=split)
    return df

def add_blank_line_after_sent(filename):
    filepath = os.path.join(os.getcwd(), '..', 'Data', 'Final', filename)
    content = open(filepath, encoding='utf-8').readlines()
    content = [x.strip() for x in content]
    print("Number of lines: ", len(content))

    curr_sent_id = 1
    lines = []
    for i, line in enumerate(content):
        if 'esp' in line.split('\t')[0] and 'eng' in content[i-1].split('\t')[0]:
            curr_sent_id = 1
        if int(line.split('\t')[1]) > curr_sent_id:
            curr_sent_id = int(line.split('\t')[1])
            lines.append("")
        lines.append(line)
    print("Number of lines: ", len(lines))

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in lines:
            f.write("%s\n" % item)
    print('Saved as ', filepath)

if __name__ == "__main__":
    # preprocess_17_spanish('es-ud-train.conllu')
    # df = load_17_spanish('train')

    # df = load_05('train')
    # df1 = load_09_english_pt('test ood') # train, val, test, test ood
    # df = load_09_english_ud('test')
    # df2 = load_09_spanish('train')  # train, val, test
    # df3 = load_09_spanish('dev')
    # df4 = load_09_spanish('test')

    # add_blank_line_after_sent('eng09-esp09_dev.txt')
    print('Done.')
