import pandas as pd
import numpy as np
from collections import defaultdict
from Bio import SeqIO

# parameters
selected_gos = [
    "GO:0007165",  # signal transduction
    "GO:0006468",  # protein phosphorylation
    "GO:0008284",  # positive regulation of cell population proliferation
    "GO:0043066"   # negative regulation of apoptotic process
]

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PAD_ID = len(AMINO_ACIDS)  # 20
SEQ_LEN = 1104
NUM_CLASSES = len(selected_gos)

aa_to_int = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
go_to_idx = {go: i for i, go in enumerate(selected_gos)}

# load annotations and sequences
annot_df = pd.read_csv("data/uniprot_sprot_exp.txt", sep="\t", header=None, names=["protein", "go", "ont"])
bp_df = annot_df[annot_df["ont"] == "P"]
seqs = SeqIO.to_dict(SeqIO.parse("data/uniprot_sprot_exp.fasta", "fasta"))

# build multilabel ontology vector for each protein
protein_to_labels = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.int32))
for _, row in bp_df.iterrows():
    go = row["go"]
    pid = row["protein"]
    if go in go_to_idx:
        protein_to_labels[pid][go_to_idx[go]] = 1

# encode 
def encode_sequence(seq, aa_map, seq_len, pad_value):
    encoded = [aa_map.get(aa, pad_value) for aa in seq.upper() if aa in aa_map]
    encoded = encoded[:seq_len] + [pad_value] * max(0, seq_len - len(encoded))
    return encoded

X, y = [], []
for pid, record in seqs.items():
    if pid not in protein_to_labels:
        continue
    encoded = encode_sequence(str(record.seq), aa_to_int, SEQ_LEN, PAD_ID)
    X.append(encoded)
    y.append(protein_to_labels[pid])

X = np.array(X, dtype=np.int32)          # shape: (N, SEQ_LEN)
y = np.array(y, dtype=np.int32)          # shape: (N, NUM_CLASSES)

np.save("X.npy", X)
np.save("y.npy", y)
print(f"Saved {X.shape[0]} sequences.")
