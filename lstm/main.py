import numpy as np
import tensorflow as tf
import pandas as pd
from keras._tf_keras.keras.layers import TextVectorization
from sklearn.metrics import f1_score
from lstm import batch_predict, LSTMModel, load_and_preprocess_test_data
import json

def main():
    print("Loading Keras model...")
    try:
        keras_model = tf.keras.models.load_model('best_model_full_old.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    embedding_input_dim = None
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            embedding_input_dim = layer.input_dim
            print(f"Found embedding input dimension: {embedding_input_dim}")
            break
    
    if not embedding_input_dim:
        print("Could not find embedding layer in the model")
        return
    
    print("Loading tokenizer...")
    try:
        vectorizer_config = np.load('vectorizer_config_old.npy', allow_pickle=True).item()
        print("Vectorizer config loaded successfully")
    except Exception as e:
        print(f"Error loading vectorizer config: {e}")
        return
    
    vectorizer = TextVectorization(
        max_tokens=embedding_input_dim, 
        output_sequence_length=vectorizer_config.get('output_sequence_length', 100),
        standardize=vectorizer_config.get('standardize', 'lower_and_strip_punctuation'),
        split=vectorizer_config.get('split', 'whitespace'),
        ngrams=None,
        output_mode='int'
    )
    
    try:
        print("Loading vocabulary data...")
        vocab_data = np.load('vectorizer_vocabulary.npy', allow_pickle=True)
        vectorizer.set_vocabulary(vocab_data)
        print(f"Loaded vocabulary with {len(vocab_data)} tokens")
    except:
        try:
            if 'vocabulary' in vectorizer_config and vectorizer_config['vocabulary'] is not None:
                vocab = vectorizer_config['vocabulary']
                vectorizer.set_vocabulary(vocab)
                print(f"Set vocabulary from config with {len(vocab)} tokens")
            else:
                print("Adapting vocabulary from training data...")
                train_df = pd.read_csv('data/train.csv')
                X_train = train_df['text'].values
                vectorizer.adapt(X_train)
                print(f"Adapted vocabulary with {len(vectorizer.get_vocabulary())} tokens")
        except Exception as e:
            print(f"Error setting vocabulary: {e}")
            return
    
    model_config = keras_model.get_config()
    # print("Model Configuration:")
    # print(json.dumps(model_config, indent=2))
    vocab_size = len(vectorizer.get_vocabulary())
    print(f"Actual vocabulary size: {vocab_size}")
    
    print("Extracting model configuration...")
    
    embedding_dim = 64
    for layer in model_config['layers']:
        if layer['class_name'] == 'Embedding':
            embedding_dim = layer['config']['output_dim']
            print(f"Found embedding dimension: {embedding_dim}")
            break
    
    lstm_layers = 0
    bidirectional = False
    lstm_units = 128
    
    for layer in model_config['layers']:
        if 'lstm' in layer['class_name'].lower():
            lstm_layers += 1
            lstm_units = layer['config']['units']
            print(f"Found LSTM layer with {lstm_units} units")
        elif 'bidirectional' in layer['class_name'].lower():
            lstm_layers += 1
            bidirectional = True
            if 'layer' in layer['config'] and 'config' in layer['config']['layer']:
                lstm_units = layer['config']['layer']['config']['units']
                print(f"Found Bidirectional LSTM layer with {lstm_units} units")
    
    dropout_rate = 0.2
    for layer in model_config['layers']:
        if 'dropout' in layer['class_name'].lower():
            dropout_rate = layer['config']['rate']
            print(f"Found dropout rate: {dropout_rate}")
            break
    
    num_classes = 3
    for layer in reversed(model_config['layers']):
        if layer['class_name'] == 'Dense':
            num_classes = layer['config']['units']
            print(f"Found output classes: {num_classes}")
            break
    
    print("\nCreating from-scratch model with configuration:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - LSTM units: {lstm_units}")
    print(f"  - LSTM layers: {lstm_layers}")
    print(f"  - Bidirectional: {bidirectional}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Number of classes: {num_classes}")
    
    from_scratch_model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units, 
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )
    
    print("Transferring weights from Keras model to from-scratch model...")
    from_scratch_model.load_weights_from_keras(keras_model)
    
    print("Loading test data...")
    X_test, y_test = load_and_preprocess_test_data('data/test.csv')
    
    print("Making predictions...")
    
    X_test_vec = vectorizer(X_test).numpy()
    print(f"Test data shape after vectorization: {X_test_vec.shape}")
    
    keras_preds = np.argmax(keras_model.predict(X_test_vec), axis=1)
    
    from_scratch_preds = np.argmax(batch_predict(from_scratch_model, vectorizer, X_test), axis=1)
    
    keras_f1 = f1_score(y_test, keras_preds, average='macro')
    from_scratch_f1 = f1_score(y_test, from_scratch_preds, average='macro')
    
    print(f"\nKeras Model F1 Score: {keras_f1}")
    print(f"From Scratch Model F1 Score: {from_scratch_f1}")
    
    match_percentage = np.mean(keras_preds == from_scratch_preds) * 100
    print(f"Prediction Match Percentage: {match_percentage}%")
    
    print("\nDetailed prediction comparison:")
    print(f"Total test samples: {len(X_test)}")
    matched = np.sum(keras_preds == from_scratch_preds)
    print(f"Matched predictions: {matched} / {len(X_test)}")
    
    mismatches = np.where(keras_preds != from_scratch_preds)[0]
    if len(mismatches) > 0:
        print(f"\nFound {len(mismatches)} mismatched predictions.")
        num_samples = min(5, len(mismatches))
        print(f"Showing {num_samples} examples:")
        
        for i in range(num_samples):
            idx = mismatches[i]
            print(f"Example {i+1}:")
            print(f"Text: {X_test[idx][:100]}..." if len(X_test[idx]) > 100 else f"Text: {X_test[idx]}")
            print(f"True label: {y_test[idx]}")
            print(f"Keras prediction: {keras_preds[idx]}")
            print(f"From-scratch prediction: {from_scratch_preds[idx]}")
            print()
    
    print("\nDebug: Comparing raw prediction probabilities...")
    keras_raw = keras_model.predict(X_test_vec[:5])
    scratch_raw = batch_predict(from_scratch_model, vectorizer, X_test[:5])
    
    # for i in range(5):
    #     print(f"Sample {i+1}:")
    #     print(f"Keras raw:    {keras_raw[i]}")
    #     print(f"Scratch raw:  {scratch_raw[i]}")
    #     print(f"Difference:   {np.abs(keras_raw[i] - scratch_raw[i])}")
    #     print()

if __name__ == "__main__":
    main()