import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight # Import this

# --- Load preprocessed data ---
X = np.load("training_data/X.npy")
y = np.load("training_data/y.npy")
num_classes = len(np.unique(y))
SEQ_LEN = X.shape[1]
VOCAB_SIZE = 21 # 20 AAs + 1 for padding (0)

# --- VERIFY CLASS IMBALANCE (as suggested above) ---
unique_labels, counts = np.unique(y, return_counts=True)
print("\n--- Class Distribution ---")
print(dict(zip(unique_labels, counts)))
print("-" * 25)

# --- Train/val split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True # Stratify is important!
)

# --- SOLUTION: CALCULATE CLASS WEIGHTS ---
# This will create weights that are inversely proportional to the class frequencies
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print("\n--- Class Weights ---")
print(class_weights_dict)
print("-" * 25)


# --- Define Model (No changes needed here) ---
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, mask_zero=True, input_length=SEQ_LEN),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # Let's try a slightly higher LR

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()


# --- Train with class_weight ---
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights_dict # <--- THE CRITICAL FIX
)

# --- Evaluate ---
print("\n--- Classification Report ---")
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_val

print(classification_report(y_true, y_pred, digits=4))
