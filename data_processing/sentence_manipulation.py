import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

universal_srl_tagset = ['A0','A1','A2','A3','A4']

class Node:
    def __init__(self, idx, head, word=None):
        self.idx = idx
        self.head = head
        self.word = word
        self.children = []

class Tree_Manipulation:
    def __init__(self, sent_id, df):
        self.root = None
        self.sent_id = sent_id
        self.subdf = df[df['Sent ID'] == sent_id]
        self.ids = self.subdf['ID'].values
        self.words = self.subdf['Form'].values
        self.parse_heads = self.subdf['Parse Head'].values
        self.predicates = self.subdf['Is Predicate'].values

        self.srl_tags = []
        for col_idx in range(9, self.subdf.shape[1]):  # Loop over all SRL columns
            col_vals = self.subdf.iloc[:, col_idx]
            if any(col_vals.notna()):
                self.srl_tags.append(col_vals.values)

    def build_tree_for_sent(self):
        # Init tree and stack
        self.root = Node(0, None, 'root')
        to_be_considered = [self.root]

        while to_be_considered != []:
            curr_node = to_be_considered.pop()
            for i in range(len(self.ids)):
                if self.parse_heads[i] == curr_node.idx:
                    newnode = Node(self.ids[i], self.parse_heads[i], self.words[i])
                    curr_node.children.append(newnode)
                    to_be_considered.append(newnode)

        return self.root

    def find_node_in_tree(self, node, id):
        if node.idx == id:
            return node

        for child in node.children:
            found = self.find_node_in_tree(child, id)
            if found is not None:
                return found

        return None

    def find_children(self, parent_id):
        parent = self.find_node_in_tree(self.root, parent_id)
        children_ids = [parent.idx]
        to_be_considered = [parent]

        while to_be_considered != []:
            curr_node = to_be_considered.pop()

            for child in curr_node.children:
                children_ids.append(child.idx)
                to_be_considered.append(child)

        return sorted(children_ids)

    def change_srl_to_bio(self):
        bio_srl_tags = []
        for j in range(len(self.srl_tags)):
            logging.debug(f"SRL Row {j}")
            srl_tags = list(self.srl_tags[j])
            bio_srl = srl_tags[:]

            # Mark the predicate for that column as 'V'
            bio_srl[[i for i, n in enumerate(self.predicates) if n == 'Y'][j]] = 'B-V'
            if bio_srl.count('B-V') > 1:
                logging.warning("Multiple word predicates.")

            for i in range(len(srl_tags)):
                if srl_tags[i] != 'O':
                    children = self.find_children(i+1)
                    tag = srl_tags[i]
                    logging.debug(f"tag: {tag} children: {children}")

                    # To maintain the same tagset in English and Spanish, we will throw away all tags like A0-LOC and keep tags like A0
                    if tag in universal_srl_tagset:
                        new_tags = ['B-' + str(tag)] + (len(children) - 1) * ['I-' + str(tag)]
                        if set(bio_srl[children[0]-1:children[0]+len(new_tags)-1]) > {'O', tag}:
                            logging.warning(f"Overwriting tags: sent {self.sent_id}, column {j}, tag id {i} for {srl_tags[i]}")
                            logging.info(f"new tags: {new_tags}")
                            logging.info(f"replace indices: {children[0] - 1, children[0] + len(new_tags) - 1} ; text is {bio_srl[children[0] - 1:children[0] + len(new_tags) - 1]}")
                            logging.info(f"len bio_srl: {len(bio_srl[children[0] - 1:children[0] + len(new_tags) - 1])}; len new tags: {len(new_tags)}")
                        bio_srl[children[0]-1:children[0]+len(new_tags)-1] = new_tags

            if 'B-V' not in bio_srl:
                logging.warning(f"'B-V' not in SRL col {j} for Sentence {self.sent_id}")
            bio_srl_tags.append(bio_srl)
        return bio_srl_tags
