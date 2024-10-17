import os
import math

import numpy as np
import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

#from .data_utils import orientation, aa_to_id

class IntGoDataset(Dataset):
    def __init__(self, root='./data/swiss-prot/', random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        #p_go_domain_file = os.path.join(root, 'p_go_domain.tsv')
        p_go_domain_file = os.path.join(root, 'p_go_domain_withneg.tsv')
        #p_go_domain_file = os.path.join(root, 'p_go_domain_withneg_dombalance.tsv')
        domain_map_file = os.path.join(root, 'domain_mapper.p')
        go_map_file = os.path.join(root, 'go_mapper.p')

        with open(domain_map_file, 'rb') as f:
            self.domain_mapper = pickle.load(f)
        with open(go_map_file, 'rb') as f:
            self.go_mapper = pickle.load(f)
        
        # Load the interpro2go file.
        df_p_domain_go = pd.read_csv(p_go_domain_file, sep='\t', header=0)

        # 使用 pickle 加载
        #with open('./data/interpro/text_label.pkl', 'rb') as f:
        #    text_label = pickle.load(f)
        with open('./data/interpro/text.pkl', 'rb') as f:
            text_embeddings = pickle.load(f)

        interpro2go = []
        for x in df_p_domain_go.to_dict('records'):
            if x['domain'] in text_embeddings:
                #label, text_embedding = text_label[x['domain']]
                label = 8 # the maximum is 7, 8 represent none
                text_embedding = text_embeddings[x['domain']]
            else:
                print(f"warning: {x['domain']} is not in text embeddings!")
                label = 8 # the maximum is 7, 8 represent none
                text_embedding = np.zeros(768)
            interpro2go.append((self.domain_mapper[x['domain']], self.go_mapper[x['goterm']], x['p_go_domain'], label, text_embedding))
        
        #interpro2go = [(self.domain_mapper[x['domain']], self.go_mapper[x['goterm']], x['p_go_domain']) for x in df_p_domain_go.to_dict('records')]
        self.random_state.shuffle(interpro2go)

        if split == 'train':
            self.data = interpro2go
        elif split == 'valid':
            self.data = interpro2go[:math.floor(len(interpro2go)*0.01)]
        elif split == 'test':
            self.data = interpro2go[-math.floor(len(interpro2go)*0.01):]
        print(f"{split} dataset, total {len(self.data)} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        domain_id, go_id, p, label, text_embedding = self.data[idx]

        data = Data(domain_id = torch.LongTensor([domain_id]),    # [1]
                    go_id = torch.LongTensor([go_id]),    # [1]
                    p = torch.FloatTensor([p]),     # [1]
                    label = torch.FloatTensor([label]),     # [1]
                    text_embedding = torch.FloatTensor(text_embedding).unsqueeze(0)    # [1, 768]
                   )

        return data
