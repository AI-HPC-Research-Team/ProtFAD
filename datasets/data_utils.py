import numpy as np
import pickle
from sklearn.preprocessing import normalize

# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i, acid in enumerate(aa):
    aa_to_id[acid] = i

def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)

# Load the fasta file.
def load_fasta(path, protein_list=None):
    """
    # load the protein_name and the corresponding amino acid sequence from the fasta file
    # the amino acid sequence is converted to a list of integers
    """
    protein_seqs = []
    with open(path, 'r') as f:
        protein_name = ''
        for line in f:
            if line.startswith('>'):
                protein_name = line.rstrip()[1:]
            else:
                if protein_list is not None and protein_name not in protein_list:
                    continue
                amino_chain = line.rstrip()
                amino_ids = []
                for amino in amino_chain:
                    amino_ids.append(aa_to_id[amino])
                protein_seqs.append((protein_name, np.array(amino_ids)))

    return protein_seqs

# Load the domain file
def load_domain(path):
    with open(path, 'rb') as f:
        domain_embeddings = pickle.load(f)
    return domain_embeddings
