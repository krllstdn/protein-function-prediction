import numpy as np
from models import cnn_model_1, cnn_inception, transformer_model, lstm_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


X = np.load("X.npy")
y = np.load("y.npy")


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 of 90% = 18%

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# model = cnn_model_1(X, y)
# model = cnn_inception(X, y)
model = lstm_model(X, y)


callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=50, callbacks=callbacks)


test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}, Test auc: {test_auc}, Test accuracy: {test_acc}")

model.save(f"inception-0.77.keras")

