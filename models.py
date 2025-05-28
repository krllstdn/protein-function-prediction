from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from keras_nlp.layers import TransformerEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def cnn_model_1(X, y):
    model = Sequential([
        Embedding(input_dim=21, output_dim=128, input_length=X.shape[1]),
        Conv1D(128, kernel_size=9, activation='relu', padding='same'),
        Dropout(0.3),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y.shape[1], activation='sigmoid')  # sigmoid for multilabel
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', 'AUC'])
    return model


def cnn_inception(X, y):
    input_layer = Input(shape=(X.shape[1],))
    x = Embedding(input_dim=21, output_dim=128, input_length=X.shape[1])(input_layer)

    # Multi-kernel convs (inception-like)
    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    conv5 = Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    conv9 = Conv1D(128, kernel_size=9, activation='relu', padding='same')(x)

    x = Concatenate()([conv3, conv5, conv9])
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(y.shape[1], activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', 'AUC'])

    return model

def transformer_model(X, y):
    model = Sequential([
        Embedding(input_dim=21, output_dim=64, input_length=X.shape[1]),
        TransformerEncoder(
            num_heads=2,
            intermediate_dim=128,
            dropout=0.1,
            activation="relu"
        ),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(y.shape[1], activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', 'AUC'])

    return model


def lstm_model(X, y):
    model = Sequential([
        Embedding(input_dim=21, output_dim=128, input_length=X.shape[1]),
        LSTM(64, return_sequences=True, recurrent_dropout=0.2),
        BatchNormalization(),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(y.shape[1], activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy', "AUC"])

    return model

