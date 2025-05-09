import numpy as np
from keras._tf_keras.keras.layers import TextVectorization, Embedding, LSTM, Bidirectional, Dropout, Dense
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import SparseCategoricalCrossentropy
from keras._tf_keras.keras.metrics import SparseCategoricalAccuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_nusax_sentiment():
    df = pd.read_csv('data/train.csv')
    return df

def preprocess_data(df):
    X = df['text'].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'].values)
    
    print(f"Original unique labels: {np.unique(df['label'].values)}")
    print(f"Encoded unique labels: {np.unique(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def create_tokenizer(X_train, max_tokens=10000, output_sequence_length=100):
    vectorizer = TextVectorization(
        max_tokens=max_tokens, 
        output_sequence_length=output_sequence_length
    )
    vectorizer.adapt(X_train)
    return vectorizer

def create_lstm_model(vocab_size, embedding_dim=64, lstm_units=128, lstm_layers=1, 
                      bidirectional=True, dropout_rate=0.2, num_classes=3):
    model = Sequential()
    
    model.add(Embedding(vocab_size, embedding_dim))
    
    for i in range(lstm_layers - 1):
        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        else:
            model.add(LSTM(lstm_units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units)))
    else:
        model.add(LSTM(lstm_units))
    
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )
    
    return model

def train_model(model, X_train_vec, y_train, X_val_vec, y_val, epochs=10, batch_size=32):
    y_train = np.asarray(y_train).astype('int32')
    y_val = np.asarray(y_val).astype('int32')
    
    history = model.fit(
        X_train_vec, y_train,
        validation_data=(X_val_vec, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

def evaluate_model(model, X_test_vec, y_test):
    y_test = np.asarray(y_test).astype('int32')
    
    y_pred = np.argmax(model.predict(X_test_vec), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    return f1, report

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_experiment(X_train, X_val, X_test, y_train, y_val, y_test, 
                   lstm_layers=1, lstm_units=128, bidirectional=True, 
                   epochs=10, batch_size=32, num_classes=3):
    
    vectorizer = create_tokenizer(X_train)
    vocab_size = len(vectorizer.get_vocabulary()) + 2 
    
    X_train_vec = vectorizer(X_train)
    X_val_vec = vectorizer(X_val)
    X_test_vec = vectorizer(X_test)
    
    model = create_lstm_model(
        vocab_size=vocab_size,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        num_classes=num_classes
    )
    
    history = train_model(
        model, X_train_vec, y_train, 
        X_val_vec, y_val, 
        epochs=epochs,
        batch_size=batch_size
    )
    
    f1, report = evaluate_model(model, X_test_vec, y_test)
    
    model_name = f"lstm_{lstm_layers}layer_{'bi' if bidirectional else 'uni'}_{lstm_units}units"
    model.save_weights(f"{model_name}.weights.h5")
    
    return model, vectorizer, history, f1, report, model_name

if __name__ == "__main__":
    df = load_nusax_sentiment()
    
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = preprocess_data(df)
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    layer_variations = [1, 2, 3]
    layer_results = []
    
    for layers in layer_variations:
        print(f"\nExperiment with {layers} LSTM layers:")
        model, vectorizer, history, f1, report, model_name = run_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            lstm_layers=layers,
            num_classes=num_classes
        )
        
        layer_results.append({
            'layers': layers,
            'f1_score': f1,
            'model': model,
            'history': history,
            'model_name': model_name
        })
        
        print(f"F1 Score: {f1}")
        print(report)
        plot_history(history)
    
    unit_variations = [64, 128, 256]
    unit_results = []
    
    for units in unit_variations:
        print(f"\nExperiment with {units} LSTM units:")
        model, vectorizer, history, f1, report, model_name = run_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            lstm_units=units,
            num_classes=num_classes
        )
        
        unit_results.append({
            'units': units,
            'f1_score': f1,
            'model': model,
            'history': history,
            'model_name': model_name
        })
        
        print(f"F1 Score: {f1}")
        print(report)
        plot_history(history)
    
    direction_variations = [True, False]  
    direction_results = []
    
    for bidirectional in direction_variations:
        direction_name = "Bidirectional" if bidirectional else "Unidirectional"
        print(f"\nExperiment with {direction_name} LSTM:")
        model, vectorizer, history, f1, report, model_name = run_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            bidirectional=bidirectional,
            num_classes=num_classes
        )
        
        direction_results.append({
            'direction': direction_name,
            'f1_score': f1,
            'model': model,
            'history': history,
            'model_name': model_name
        })
        
        print(f"F1 Score: {f1}")
        print(report)
        plot_history(history)
    
    best_model_idx = np.argmax([r['f1_score'] for r in layer_results + unit_results + direction_results])
    all_results = layer_results + unit_results + direction_results
    best_model = all_results[best_model_idx]['model']
    best_model_name = all_results[best_model_idx]['model_name']
    
    print(f"\nBest model: {best_model_name} with F1 score: {all_results[best_model_idx]['f1_score']}")
    
    best_model.save("best_model_full.h5")
    
    np.save("vectorizer_config.npy", vectorizer.get_config())
    
    np.save("label_encoder_classes.npy", label_encoder.classes_)