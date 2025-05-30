import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.layers import TextVectorization, LSTM, Bidirectional, Dropout, Dense

class Embedding:
    def __init__(self, input_dim, output_dim, weights=None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.normal(0, 1, (input_dim, output_dim))
    
    def forward(self, x):
        x_clipped = np.clip(x, 0, self.weights.shape[0] - 1)
        return np.take(self.weights, x_clipped, axis=0)

class LSTM:
    def __init__(self, units, return_sequences=False, weights=None, input_dim=None):
        self.units = units
        self.return_sequences = return_sequences
        
        if weights is not None:
            self.W_i, self.U_i, self.b_i = weights[0]
            self.W_f, self.U_f, self.b_f = weights[1]
            self.W_c, self.U_c, self.b_c = weights[2]
            self.W_o, self.U_o, self.b_o = weights[3]
        else:
            if input_dim is None:
                print("Warning: input_dim not specified for LSTM, weights will be initialized during first forward pass")
                self.input_dim = None
                self.W_i = self.U_i = self.b_i = None
                self.W_f = self.U_f = self.b_f = None
                self.W_c = self.U_c = self.b_c = None
                self.W_o = self.U_o = self.b_o = None
                self.weights_initialized = False
            else:
                self._initialize_weights(input_dim)
                self.weights_initialized = True
    
    def _initialize_weights(self, input_dim):
        """Initialize LSTM weights with Xavier/Glorot initialization"""
        scale_in = np.sqrt(2.0 / input_dim)
        scale_rec = np.sqrt(2.0 / self.units)
        
        self.W_i = np.random.normal(0, scale_in, (input_dim, self.units)).astype(np.float32)
        self.W_f = np.random.normal(0, scale_in, (input_dim, self.units)).astype(np.float32)
        self.W_c = np.random.normal(0, scale_in, (input_dim, self.units)).astype(np.float32)
        self.W_o = np.random.normal(0, scale_in, (input_dim, self.units)).astype(np.float32)
        
        self.U_i = np.random.normal(0, scale_rec, (self.units, self.units)).astype(np.float32)
        self.U_f = np.random.normal(0, scale_rec, (self.units, self.units)).astype(np.float32)
        self.U_c = np.random.normal(0, scale_rec, (self.units, self.units)).astype(np.float32)
        self.U_o = np.random.normal(0, scale_rec, (self.units, self.units)).astype(np.float32)
        
        self.b_i = np.zeros((self.units,), dtype=np.float32)
        self.b_f = np.ones((self.units,), dtype=np.float32)
        self.b_c = np.zeros((self.units,), dtype=np.float32)
        self.b_o = np.zeros((self.units,), dtype=np.float32)
        
        self.weights_initialized = True
    
    def sigmoid(self, x):
        x_clipped = np.clip(x, -250, 250)
        pos_mask = x_clipped >= 0
        neg_mask = ~pos_mask
        
        result = np.zeros_like(x_clipped)
        result[pos_mask] = 1.0 / (1.0 + np.exp(-x_clipped[pos_mask]))
        exp_x = np.exp(x_clipped[neg_mask])
        result[neg_mask] = exp_x / (1.0 + exp_x)
        
        return result
    
    def _orthogonal_init(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        x = x.astype(np.float32)
        if not hasattr(self, 'weights_initialized') or not self.weights_initialized:
            self._initialize_weights(input_dim)
        
        self.W_i = self.W_i.astype(np.float32)
        self.U_i = self.U_i.astype(np.float32)
        self.b_i = self.b_i.astype(np.float32)
        self.W_f = self.W_f.astype(np.float32)
        self.U_f = self.U_f.astype(np.float32)
        self.b_f = self.b_f.astype(np.float32)
        self.W_c = self.W_c.astype(np.float32)
        self.U_c = self.U_c.astype(np.float32)
        self.b_c = self.b_c.astype(np.float32)
        self.W_o = self.W_o.astype(np.float32)
        self.U_o = self.U_o.astype(np.float32)
        self.b_o = self.b_o.astype(np.float32)
        
        h_t = np.zeros((batch_size, self.units), dtype=np.float32)
        c_t = np.zeros((batch_size, self.units), dtype=np.float32)
        
        if self.return_sequences:
            h_seq = np.zeros((batch_size, seq_len, self.units), dtype=np.float32)
        
        for t in range(seq_len):
            x_t = x[:, t, :].astype(np.float32)
            
            i_t = self.sigmoid(np.dot(x_t, self.W_i) + np.dot(h_t, self.U_i) + self.b_i)
            f_t = self.sigmoid(np.dot(x_t, self.W_f) + np.dot(h_t, self.U_f) + self.b_f)
            c_tilde = self.tanh(np.dot(x_t, self.W_c) + np.dot(h_t, self.U_c) + self.b_c)
            
            # if t == 0 and batch_size > 0:
            #     print(f"Timestep {t}, Sample 0 - i_t: {i_t[0, :5]}, f_t: {f_t[0, :5]}, c_tilde: {c_tilde[0, :5]}")
            
            c_t = f_t * c_t + i_t * c_tilde
            
            o_t = self.sigmoid(np.dot(x_t, self.W_o) + np.dot(h_t, self.U_o) + self.b_o)
            h_t = o_t * self.tanh(c_t)
            
            if self.return_sequences:
                h_seq[:, t, :] = h_t
        
        return h_seq if self.return_sequences else h_t

class BidirectionalLSTM:
    def __init__(self, units, return_sequences=False, weights=None, input_dim=None):
        self.units = units
        self.return_sequences = return_sequences
        
        if weights is not None:
            self.forward_lstm = LSTM(units, return_sequences=True, weights=weights[0])
            self.backward_lstm = LSTM(units, return_sequences=True, weights=weights[1])
        else:
            self.forward_lstm = LSTM(units, return_sequences=True, input_dim=input_dim)
            self.backward_lstm = LSTM(units, return_sequences=True, input_dim=input_dim)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        forward_out = self.forward_lstm.forward(x)
        
        x_reversed = x[:, ::-1, :] 
        backward_out_reversed = self.backward_lstm.forward(x_reversed)
        backward_out = backward_out_reversed[:, ::-1, :] if self.return_sequences else backward_out_reversed
        
        out = np.concatenate([forward_out, backward_out], axis=2)
        
        if not self.return_sequences and out.shape[1] > 1:
            out = out[:, -1, :]
        
        return out
    
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
    
    def forward(self, x, training=False):
        if not training or self.rate == 0:
            return x
        
        return x

class Dense:
    def __init__(self, units, activation=None, weights=None, input_dim=None):
        self.units = units
        self.activation = activation
        
        if weights is not None:
            self.weights, self.bias = weights
        else:
            if input_dim is not None:
                self._initialize_weights(input_dim)
                self.weights_initialized = True
            else:
                self.weights = None
                self.bias = None
                self.weights_initialized = False
    
    def _initialize_weights(self, input_dim):
        limit = np.sqrt(6.0 / (input_dim + self.units))
        
        self.weights = np.random.uniform(-limit, limit, (input_dim, self.units)).astype(np.float32)
        self.bias = np.zeros((self.units,), dtype=np.float32)
        
        self.weights_initialized = True
    
    def softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(np.clip(x_shifted, -500, 500))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def forward(self, x):
        if self.weights is None:
            if not hasattr(self, 'weights_initialized') or not self.weights_initialized:
                input_dim = x.shape[-1]
                self._initialize_weights(input_dim)
        
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'softmax':
            return self.softmax(output)
        elif self.activation == 'relu':
            return np.maximum(0, output)
        elif self.activation == 'tanh':
            return np.tanh(output)
        else:
            return output

class LSTMModel:
    def __init__(self, vocab_size, embedding_dim, lstm_units, lstm_layers, bidirectional, dropout_rate, num_classes):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        self.layers = []
        self.initialize_layers()
    
    def initialize_layers(self):
        self.layers.append(('embedding', Embedding(self.vocab_size, self.embedding_dim)))
        
        for i in range(self.lstm_layers - 1):
            input_dim = self.embedding_dim if i == 0 else (self.lstm_units * 2 if self.bidirectional else self.lstm_units)
            if self.bidirectional:
                self.layers.append((f'bidirectional_lstm_{i}', BidirectionalLSTM(self.lstm_units, return_sequences=True, input_dim=input_dim)))
            else:
                self.layers.append((f'lstm_{i}', LSTM(self.lstm_units, return_sequences=True, input_dim=input_dim)))
            self.layers.append((f'dropout_{i}', Dropout(self.dropout_rate)))
        
        input_dim = self.embedding_dim if self.lstm_layers == 1 else (self.lstm_units * 2 if self.bidirectional else self.lstm_units)
        if self.bidirectional:
            self.layers.append((f'bidirectional_lstm_{self.lstm_layers-1}', BidirectionalLSTM(self.lstm_units, return_sequences=False, input_dim=input_dim)))
        else:
            self.layers.append((f'lstm_{self.lstm_layers-1}', LSTM(self.lstm_units, return_sequences=False, input_dim=input_dim)))
        self.layers.append((f'dropout_{self.lstm_layers-1}', Dropout(self.dropout_rate)))
        
        dense_input_dim = self.lstm_units * 2 if self.bidirectional else self.lstm_units
        self.layers.append(('dense', Dense(self.num_classes, activation='softmax', input_dim=dense_input_dim)))
        
    def forward(self, x, training=False):
        for name, layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        # print(f"After layer {name}, output shape: {x.shape}, sample values: {x[0, :5] if x.ndim == 2 else x[0, -1, :5] if x.ndim == 3 else x[0, :5]}")
        return x
    
    def load_weights_from_keras(self, keras_model):
        print("Loading weights from Keras model...")
        
        embedding_layer = None
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                embedding_layer = layer
                break
        
        if embedding_layer:
            embedding_weights = embedding_layer.get_weights()[0]
            self.layers[0][1].weights = embedding_weights
            print(f"Loaded embedding weights: {embedding_weights.shape}")
        
        keras_lstm_layers = []
        for i, layer in enumerate(keras_model.layers):
            if (isinstance(layer, tf.keras.layers.LSTM) or 
                isinstance(layer, tf.keras.layers.Bidirectional)):
                keras_lstm_layers.append((i, layer))
        
        print(f"Found {len(keras_lstm_layers)} LSTM layers in Keras model")
        
        scratch_lstm_idx = 0
        for keras_idx, keras_layer in keras_lstm_layers:
            for i, (name, layer) in enumerate(self.layers):
                if ('lstm' in name or 'bidirectional' in name) and str(scratch_lstm_idx) in name:
                    if isinstance(keras_layer, tf.keras.layers.Bidirectional):
                        self._load_bidirectional_weights(keras_layer, layer)
                    else:
                        self._load_lstm_weights(keras_layer, layer)
                    scratch_lstm_idx += 1
                    break
        
        dense_layer = None
        for layer in reversed(keras_model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                dense_layer = layer
                break
        
        if dense_layer:
            dense_weights = dense_layer.get_weights()
            self.layers[-1][1].weights = dense_weights[0]
            self.layers[-1][1].bias = dense_weights[1]
            print(f"Loaded dense weights: {dense_weights[0].shape}, bias: {dense_weights[1].shape}")
    
    def _load_lstm_weights(self, keras_lstm, scratch_lstm):
        weights = keras_lstm.get_weights()
        
        if len(weights) != 3:
            print(f"Warning: Expected 3 weight matrices, got {len(weights)}")
            return
        
        kernel, recurrent_kernel, bias = weights
        units = self.lstm_units
        
        W_i = kernel[:, :units]
        W_f = kernel[:, units:2*units]
        W_c = kernel[:, 2*units:3*units]
        W_o = kernel[:, 3*units:]
        
        U_i = recurrent_kernel[:, :units]
        U_f = recurrent_kernel[:, units:2*units]
        U_c = recurrent_kernel[:, 2*units:3*units]
        U_o = recurrent_kernel[:, 3*units:]
        
        b_i = bias[:units]
        b_f = bias[units:2*units]
        b_c = bias[2*units:3*units]
        b_o = bias[3*units:]
        
        scratch_lstm.W_i, scratch_lstm.U_i, scratch_lstm.b_i = W_i, U_i, b_i
        scratch_lstm.W_f, scratch_lstm.U_f, scratch_lstm.b_f = W_f, U_f, b_f
        scratch_lstm.W_c, scratch_lstm.U_c, scratch_lstm.b_c = W_c, U_c, b_c
        scratch_lstm.W_o, scratch_lstm.U_o, scratch_lstm.b_o = W_o, U_o, b_o
        scratch_lstm.weights_initialized = True
        
        print(f"Loaded LSTM weights - kernel: {kernel.shape}, recurrent: {recurrent_kernel.shape}, bias: {bias.shape}")
    
    def _load_bidirectional_weights(self, keras_bidirectional, scratch_bidirectional):
        """Load weights from Keras Bidirectional LSTM to scratch Bidirectional LSTM"""
        forward_layer = keras_bidirectional.forward_layer
        backward_layer = keras_bidirectional.backward_layer
        
        self._load_lstm_weights(forward_layer, scratch_bidirectional.forward_lstm)
        
        self._load_lstm_weights(backward_layer, scratch_bidirectional.backward_lstm)
        
        print("Loaded Bidirectional LSTM weights")

def batch_predict(model, vectorizer, X, batch_size=32):
    """Make predictions in batches to handle memory efficiently"""
    n_samples = len(X)
    n_batches = int(np.ceil(n_samples / batch_size))
    predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_X = X[start_idx:end_idx]
        
        batch_X_vec = vectorizer(batch_X).numpy()
        
        batch_preds = model.forward(batch_X_vec, training=False)
        predictions.append(batch_preds)
    
    return np.vstack(predictions)

def load_and_preprocess_test_data(test_csv_path):
    """Load and preprocess test data"""
    print(f"Loading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    
    X_test = test_df['text'].values
    
    if 'label' in test_df.columns:
        print("Found 'label' column in test data")
        
        try:
            label_encoder_classes = np.load("label_encoder_classes_old.npy", allow_pickle=True)
            print(f"Loaded label encoder classes: {label_encoder_classes}")
            
            label_encoder = LabelEncoder()
            label_encoder.classes_ = label_encoder_classes
            
            y_test = label_encoder.transform(test_df['label'].values)
            print(f"Unique encoded labels in test data: {np.unique(y_test)}")
            
        except (FileNotFoundError, IOError):
            print("Label encoder classes file not found. Creating new encoder.")
            label_encoder = LabelEncoder()
            y_test = label_encoder.fit_transform(test_df['label'].values)
            print(f"Unique labels in test data: {np.unique(test_df['label'].values)}")
            print(f"Unique encoded labels: {np.unique(y_test)}")
    else:
        print("No 'label' column found in test data. Using dummy labels.")
        y_test = np.zeros(len(X_test), dtype=int)
    
    return X_test, y_test