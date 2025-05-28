

# # taken from here: https://medium.com/data-science/protein-sequence-classification-99c80d0ad2df
# codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
#
# def create_dict(codes):
#   char_dict = {}
#   for index, val in enumerate(codes):
#     char_dict[val] = index+1
#
#   return char_dict
#
# char_dict = create_dict(codes)
#
# print(char_dict)
# print("Dict Length:", len(char_dict))
#
# def integer_encoding(data):
#   """
#   - Encodes code sequence to integer values.
#   - 20 common amino acids are taken into consideration
#     and rest 4 are categorized as 0.
#   """
#   
#   encode_list = []
#   for row in data['sequence'].values:
#     row_encode = []
#     for code in row:
#       row_encode.append(char_dict.get(code, 0))
#     encode_list.append(np.array(row_encode))
#   
#   return encode_list
#
#
#
# train_encode = integer_encoding(train_sm) 
# val_encode = integer_encoding(val_sm) 
# test_encode = integer_encoding(test_sm)
#
#
# from keras.preprocessing.sequence import pad_sequences
#
# # padding sequences
# max_length = 100
# train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
# val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
# test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')
#
# train_pad.shape, val_pad.shape, test_pad.shape 

# ----



# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from Bio import SeqIO
#
# # === Load annotations ===
# annot_df = pd.read_csv("data/uniprot_sprot_exp.txt", sep="\t", header=None, names=["protein", "go", "ont"])
# bp_df = annot_df[annot_df["ont"] == "P"]
#
# selected_gos = ["GO:0007165", "GO:0006468", "GO:0008284", "GO:0043066"]
# seqs = SeqIO.to_dict(SeqIO.parse("data/uniprot_sprot_exp.fasta", "fasta"))
#
# # === GO → proteins ===
# go_to_proteins = defaultdict(set)
# for _, row in bp_df.iterrows():
#     go_to_proteins[row["go"]].add(row["protein"])
#
# # === Export selected GO sequences ===
# for go in selected_gos:
#     proteins = go_to_proteins[go]
#     out_seqs = [seqs[pid] for pid in proteins if pid in seqs]
#     SeqIO.write(out_seqs, f"training_data/{go.replace(':', '_')}.fasta", "fasta")
#     print(f"Exported {len(out_seqs)} sequences for {go}")
#
# fasta_files = [f"training_data/{go.replace(':', '_')}.fasta" for go in selected_gos]
#
# # === Define encoding ===
# AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# PAD_ID = len(AMINO_ACIDS)  # 20
# aa_to_int = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
#
# SEQ_LEN = 1104
# NUM_CLASSES = 4
# label_map = {f: i for i, f in enumerate(fasta_files)}
#
# def encode_sequence(seq, aa_map, seq_len, pad_value):
#     encoded = [aa_map.get(aa, pad_value) for aa in seq.upper() if aa in aa_map]
#     encoded = encoded[:seq_len] + [pad_value] * max(0, seq_len - len(encoded))
#     return encoded
#
# # === Encode dataset ===
# X, y = [], []
# for fasta_file in fasta_files:
#     label = label_map[fasta_file]
#     for record in SeqIO.parse(fasta_file, "fasta"):
#         encoded = encode_sequence(str(record.seq), aa_to_int, SEQ_LEN, PAD_ID)
#         X.append(encoded)
#         y.append(label)
#
# X = np.array(X, dtype=np.int32)  # shape: (N, SEQ_LEN)
# y = np.array(y, dtype=np.int32)  # shape: (N,)
#
# # === Save for training ===
# np.save("X.npy", X)
# np.save("y.npy", y)



import pandas as pd
import numpy as np
from collections import defaultdict
from Bio import SeqIO
import os

# === Load annotations ===
annot_df = pd.read_csv("data/uniprot_sprot_exp.txt", sep="\t", header=None, names=["protein", "go", "ont"])
bp_df = annot_df[annot_df["ont"] == "P"]

selected_gos = ["GO:0007165", "GO:0006468", "GO:0008284", "GO:0043066"]
seqs = SeqIO.to_dict(SeqIO.parse("data/uniprot_sprot_exp.fasta", "fasta"))

# === GO → proteins ===
go_to_proteins = defaultdict(set)
for _, row in bp_df.iterrows():
    if row["go"] in selected_gos:
        go_to_proteins[row["go"]].add(row["protein"])

# === Create non-overlapping dataset ===
X_data, y_data = [], []
label_map = {go: i for i, go in enumerate(selected_gos)}
assigned_proteins = set()

print("Creating non-overlapping dataset...")
for go in selected_gos:
    label = label_map[go]
    proteins_for_go = go_to_proteins[go]
    
    count = 0
    for protein_id in proteins_for_go:
        # Only add protein if it hasn't been assigned to another class yet
        if protein_id in seqs and protein_id not in assigned_proteins:
            X_data.append(str(seqs[protein_id].seq))
            y_data.append(label)
            assigned_proteins.add(protein_id)
            count += 1
    print(f"Assigned {count} unique sequences for {go} (label {label})")

# === Define encoding (FIX: Use 0 for padding) ===
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY' # 20 standard amino acids
# Reserve 0 for padding, map amino acids from 1 to 21
aa_to_int = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
PAD_ID = 0 
SEQ_LEN = 1104 # Ensure this is a reasonable max length for your data

def encode_sequence(seq, aa_map, seq_len, pad_value):
    encoded = [aa_map.get(aa, pad_value) for aa in seq.upper()]
    # Truncate if longer, pad if shorter
    encoded = encoded[:seq_len] + [pad_value] * max(0, seq_len - len(encoded))
    return encoded

# === Encode dataset ===
X_encoded = [encode_sequence(seq, aa_to_int, SEQ_LEN, PAD_ID) for seq in X_data]

X = np.array(X_encoded, dtype=np.int32)
y = np.array(y_data, dtype=np.int32)

print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")

# === Save for training ===
if not os.path.exists("training_data"):
    os.makedirs("training_data")
np.save("training_data/X.npy", X)
np.save("training_data/y.npy", y)




# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Masking
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from tensorflow.keras.optimizers import Adam
#
# # --- Load preprocessed data ---
# X = np.load("X.npy")          # shape: (N, SEQ_LEN, 20)
# y = np.load("y.npy")          # shape: (N,)
# num_classes = len(np.unique(y))
# SEQ_LEN = X.shape[1]
#
#
# # --- Train/val split ---
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
#
# model = Sequential([
#     Embedding(input_dim=PAD_ID + 1, output_dim=64, mask_zero=True, input_length=SEQ_LEN),
#     LSTM(256, return_sequences=True),
#     Dropout(0.3),
#     LSTM(128),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # --- Train ---
# callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
# history = model.fit(X_train, y_train,
#                     validation_data=(X_val, y_val),
#                     batch_size=32,
#                     epochs=50,
#                     callbacks=callbacks)
#
# # --- Evaluate ---
# y_pred = model.predict(X_val).argmax(axis=1)
# y_true = y_val.argmax(axis=1)
#
# print(classification_report(y_true, y_pred, digits=4))


