import pandas as pd
import logging
from data_processing.load_utils import *
logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def drop_spurious_lines(filename):
    filepath = os.path.join(DATADIR, 'Spanish', 'CoNLL_2017', filename)
    content = open(filepath, encoding='utf-8').readlines()
    content = [x.strip() for x in content]
    print("Number of lines: ", len(content))
    lines=[]
    for line in content:
        if line != '':
            if line[0] != '#' and '-' not in line.split('\t')[0]:
                lines.append(line)
    print(len(lines), " lines retained.")

    with open(filepath+'.txt', 'w', encoding='utf-8') as f:
        for item in lines:
            f.write("%s\n" % item)
    print('Saved as ', filepath+'.txt')

if __name__ == "__main__":
    split = 'train'
    drop_spurious_lines(f'es-ud-{split}.conllu')

    df = load_17_spanish(split, raw=True)
    df['Domain'] = 'esp_conll17'
    df = df.reindex([*[df.columns[-1]], *df.columns[:-1]], axis=1)
    df['PennTree POS'] = '_'
    df = df.reindex(['Domain', 'Sent ID', 'ID', 'Form', 'Lemma', 'PennTree POS', 'POS', 'Parse Head', 'Parse Label'], axis=1)

    df.to_csv(os.path.join(DATADIR, 'Spanish', '2017_normalized', f'esp17_{split}.txt'), sep='\t', header=False,
                     index=False, encoding='utf-8', quoting=csv.QUOTE_NONE)
    add_blank_line_after_sent(f'esp17_{split}.txt', os.path.join('Spanish', '2017_normalized'))
    logging.info("Done.")