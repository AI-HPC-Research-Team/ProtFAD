import os
import math

import numpy as np
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .data_utils import orientation, load_domain, load_fasta

class ProteinDataset(Dataset):
    def __init__(self, root, split, random_seed=0, sub_sample=False, request_domain='any'):
        assert request_domain in ['any', 'exist', 'noexist'], "request_domain must be one of ['any', 'exist', 'noexist']"
        self.random_state = np.random.RandomState(random_seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino, seq_emb, domain = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)
        seq_emb = seq_emb.unsqueeze(0)
        domain = domain[np.newaxis, :].astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos),    # [num_nodes, num_dimensions]
                    seq_emb = seq_emb,    # [1, 1280]
                    domain = torch.from_numpy(domain),    # [1, 768]
                    #name = protein_name
                   )

        return data

class FoldDataset(Dataset):

    def __init__(self, root='./data/fold', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates', split)
        fasta_file = os.path.join(root, split+'.fasta')
        seq_emb_dir = os.path.join(root, 'seq_embeddings')
        domain_file = os.path.join(root, f"domain_embeddings.p")
        domain_dim = 768

        # Load the fasta file.
        protein_seqs = load_fasta(fasta_file)

        # Load the domain file
        domain_embeddings = load_domain(domain_file)

        fold_classes = {}
        with open(os.path.join(root, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(root, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        count = 0
        for protein_name, amino_ids in tqdm(protein_seqs, desc="process protein data"):
            # domain embeddings
            if protein_name in domain_embeddings:
                domain_poss, domain_embs, domain_ids = domain_embeddings[protein_name] # [num_domains], [num_domains, 256] [num_domains]
                domain_num = len(domain_poss)
                domain_poss = np.pad(domain_poss, [(0, 13-domain_num)])
                domain_embs = np.pad(domain_embs, [(0, 13-domain_num), (0, 0)])
                domain_ids = np.pad(domain_ids, [(0, 13-domain_num)])
            else:
                count += 1
                domain_num = 0
                domain_poss = np.zeros(13)
                domain_embs = np.zeros((13, domain_dim))
                domain_ids = np.zeros(13)

            # structure data
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            # sequence embeddings
            seq_emb = torch.load(os.path.join(seq_emb_dir, protein_name+".pt"))['mean_representations'][33]

            self.data.append((protein_name, pos, ori, amino_ids.astype(int), seq_emb, domain_num, domain_embs, domain_poss, domain_ids))

            self.labels.append(protein_folds[protein_name])
        print(f"{split} data: total {len(self.data)} proteins, {count} proteins have no domain embeddings.")    

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino, seq_emb, domain_num, domain_embs, domain_poss, domain_ids = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
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
                    pos = torch.from_numpy(pos),     # [num_nodes, num_dimensions]
                    seq_emb = seq_emb,    # [1, 1280]
                    #domain = torch.from_numpy(domain),    # [1, 768]
                    domain_num = torch.from_numpy(domain_num),    # [1,]
                    domain_embs = torch.from_numpy(domain_embs),    # [1, 16, 256]
                    domain_poss = torch.from_numpy(domain_poss),    # [1, 16]
                    domain_ids = torch.from_numpy(domain_ids),    # [1, 16]
                    name = protein_name
                    )    

        return data

class FuncDataset(Dataset):

    def __init__(self, root='./data/func', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(os.path.join(root, 'coordinates'), split)
        fasta_file = os.path.join(root, 'chain_'+split+'.fasta')
        seq_emb_dir = os.path.join(root, 'seq_embeddings')
        domain_file = os.path.join(root, f"domain_embeddings.p")
        domain_dim = 768

        # Load the fasta file.
        protein_seqs = load_fasta(fasta_file)

        # Load the domain file
        domain_embeddings = load_domain(domain_file)

        protein_functions = {}
        with open(os.path.join(root, 'chain_functions.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split(',')
                protein_functions[arr[0]] = int(arr[1])

        self.data = []
        self.labels = []
        count = 0
        for protein_name, amino_ids in tqdm(protein_seqs, desc="process protein data"):
            # domain embeddings
            if protein_name in domain_embeddings:
                domain_poss, domain_embs, domain_ids = domain_embeddings[protein_name] # [num_domains], [num_domains, 256] [num_domains]
                domain_num = len(domain_poss)
                domain_poss = np.pad(domain_poss, [(0, 20-domain_num)])
                domain_embs = np.pad(domain_embs, [(0, 20-domain_num), (0, 0)])
                domain_ids = np.pad(domain_ids, [(0, 20-domain_num)])
            else:
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
            self.labels.append(protein_functions[protein_name])
        print(f"{split} data: total {len(self.data)} proteins, {count} proteins have no domain embeddings.")    

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino, seq_emb, domain_num, domain_embs, domain_poss, domain_ids = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
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
                    domain_embs = torch.from_numpy(domain_embs),    # [1, 16, 256]
                    domain_poss = torch.from_numpy(domain_poss),    # [1, 16]
                    domain_ids = torch.from_numpy(domain_ids),    # [1, 16]
                    name = protein_name
                    )

        return data

