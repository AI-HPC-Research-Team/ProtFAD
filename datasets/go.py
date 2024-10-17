import os
import numpy as np
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .data_utils import orientation, load_fasta, load_domain

class GODataset(Dataset):

    def __init__(self, root='./data/go/', level='mf', percent=30, random_seed=0, split='train', request_domain='exist', domain_dim=768, domain_train='got'):

        assert request_domain in ['any', 'exist', 'noexist'], "request_domain must be one of ['any', 'exist', 'noexist']"
        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')
        seq_emb_dir = os.path.join(root, 'seq_embeddings')
        domain_file = os.path.join(root, f"domain_embeddings_{domain_dim}_{domain_train}.p")

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = load_fasta(fasta_file) if split != "test" else load_fasta(fasta_file, test_set)

        # Load the domain file
        domain_embeddings = load_domain(domain_file)

        self.data = []
        count = 0
        for protein_name, amino_ids in tqdm(protein_seqs, desc="process protein data"):
            # domain embdddings
            if protein_name in domain_embeddings:
                if request_domain == 'noexist': continue
                domain_poss, domain_embs, domain_ids = domain_embeddings[protein_name] # [num_domains], [num_domains, 256] [num_domains]
                domain_num = len(domain_poss)
                domain_poss = np.pad(domain_poss, [(0, 20-domain_num)])
                domain_embs = np.pad(domain_embs, [(0, 20-domain_num), (0, 0)])
                domain_ids = np.pad(domain_ids, [(0, 20-domain_num)])
            else:
                if request_domain == 'exist': continue
                count += 1
                domain_num = 0
                domain_poss = np.zeros(20)
                domain_embs = np.zeros((20, domain_dim))
                domain_ids = np.zeros(20)

            # structure data
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            
            # sequence embeddings
            seq_emb = torch.load(os.path.join(seq_emb_dir, protein_name+".pt"))['mean_representations'][33]
            
            self.data.append((protein_name, pos, ori, amino_ids.astype(int), seq_emb, domain_num, domain_embs, domain_poss, domain_ids))
        print(f"{split} data: total {len(self.data)} proteins, {count} proteins have no domain embeddings.")

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino, seq_emb, domain_num, domain_embs, domain_poss, domain_ids = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)
        seq_emb = seq_emb.unsqueeze(0)
        #domain = domain[np.newaxis, :].astype(dtype=np.float32)
        domain_num = np.array(domain_num)[np.newaxis].astype(dtype=np.int32)
        domain_embs = domain_embs[np.newaxis, :].astype(dtype=np.float32)
        domain_poss = domain_poss[np.newaxis, :].astype(dtype=np.float32)
        domain_ids = domain_ids[np.newaxis, :].astype(dtype=np.int32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos),    # [num_nodes, num_dimensions]
                    seq_emb = seq_emb,    # [1, 1280]
                    #domain = torch.from_numpy(domain),    # [1, 768]
                    domain_num = torch.from_numpy(domain_num),    # [1,]
                    domain_embs = torch.from_numpy(domain_embs),    # [1, 19, 256]
                    domain_poss = torch.from_numpy(domain_poss),    # [1, 19]
                    domain_ids = torch.from_numpy(domain_ids),    # [1, 19]
                    #name = protein_name
                   )

        return data    

