import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Embedding:
    def __init__(self, input_dim, output_dim, weights=None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.normal(0, 1, (input_dim, output_dim))
    
    def forward(self, x):
        return np.take(self.weights, x, axis=0)

class LSTM:
    def __init__(self, units, return_sequences=False, weights=None):
        self.units = units
        self.return_sequences = return_sequences
        
        if weights is not None:
            self.W_i, self.U_i, self.b_i = weights[0]
            self.W_f, self.U_f, self.b_f = weights[1]
            self.W_c, self.U_c, self.b_c = weights[2]
            self.W_o, self.U_o, self.b_o = weights[3]
        else:
            input_dim = None  
            self.W_i = self.U_i = self.b_i = None
            self.W_f = self.U_f = self.b_f = None
            self.W_c = self.U_c = self.b_c = None
            self.W_o = self.U_o = self.b_o = None
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        h_t = np.zeros((batch_size, self.units))
        c_t = np.zeros((batch_size, self.units))
        
        if self.return_sequences:
            h_seq = np.zeros((batch_size, seq_len, self.units))
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            i_t = self.sigmoid(np.dot(x_t, self.W_i) + np.dot(h_t, self.U_i) + self.b_i)
            
            f_t = self.sigmoid(np.dot(x_t, self.W_f) + np.dot(h_t, self.U_f) + self.b_f)
            
            c_tilde = np.tanh(np.dot(x_t, self.W_c) + np.dot(h_t, self.U_c) + self.b_c)
            
            c_t = f_t * c_t + i_t * c_tilde
            
            o_t = self.sigmoid(np.dot(x_t, self.W_o) + np.dot(h_t, self.U_o) + self.b_o)
            
            h_t = o_t * np.tanh(c_t)
            
            if self.return_sequences:
                h_seq[:, t, :] = h_t
        
        if self.return_sequences:
            return h_seq
        else:
            return h_t
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

class BidirectionalLSTM:
    def __init__(self, units, return_sequences=False, weights=None):
        self.units = units
        self.return_sequences = return_sequences
        
        if weights is not None:
            self.forward_lstm = LSTM(units, return_sequences=True, weights=weights[0])
            self.backward_lstm = LSTM(units, return_sequences=True, weights=weights[1])
        else:
            self.forward_lstm = LSTM(units, return_sequences=True)
            self.backward_lstm = LSTM(units, return_sequences=True)
    
    def forward(self, x):
        forward_out = self.forward_lstm.forward(x)
        
        x_reversed = x[:, ::-1, :]
        backward_out = self.backward_lstm.forward(x_reversed)
        backward_out = backward_out[:, ::-1, :]
        
        out = np.concatenate([forward_out, backward_out], axis=2)
        
        if not self.return_sequences:
            out = out[:, -1, :]
        
        return out

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
    
    def forward(self, x, training=False):
        if not training or self.rate == 0:
            return x
        
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
        return x * self.mask

class Dense:
    def __init__(self, units, activation=None, weights=None):
        self.units = units
        self.activation = activation
        
        if weights is not None:
            self.weights, self.bias = weights
        else:
            self.weights = None
            self.bias = None
    
    def forward(self, x):
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'softmax':
            return self.softmax(output)
        elif self.activation == 'relu':
            return np.maximum(0, output)
        elif self.activation == 'tanh':
            return np.tanh(output)
        else:
            return output
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

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
            if self.bidirectional:
                self.layers.append((f'bidirectional_lstm_{i}', BidirectionalLSTM(self.lstm_units, return_sequences=True)))
            else:
                self.layers.append((f'lstm_{i}', LSTM(self.lstm_units, return_sequences=True)))
            self.layers.append((f'dropout_{i}', Dropout(self.dropout_rate)))
        
        if self.bidirectional:
            self.layers.append((f'bidirectional_lstm_{self.lstm_layers-1}', BidirectionalLSTM(self.lstm_units, return_sequences=False)))
        else:
            self.layers.append((f'lstm_{self.lstm_layers-1}', LSTM(self.lstm_units, return_sequences=False)))
        self.layers.append((f'dropout_{self.lstm_layers-1}', Dropout(self.dropout_rate)))
        
        self.layers.append(('dense', Dense(self.num_classes, activation='softmax')))
    
    def forward(self, x, training=False):
        for name, layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x
    
    def load_weights_from_keras(self, keras_model):
        embedding_layer = keras_model.get_layer(index=0)
        embedding_weights = embedding_layer.get_weights()[0]
        self.layers[0][1].weights = embedding_weights
        
        layer_index = 1
        keras_layer_index = 1
        
        for i in range(self.lstm_layers):
            if self.bidirectional:
                bi_lstm_layer = keras_model.get_layer(index=keras_layer_index)
                
                forward_layer = bi_lstm_layer.forward_layer
                backward_layer = bi_lstm_layer.backward_layer
                
                forward_weights = forward_layer.get_weights()
                
                input_dim = forward_weights[0].shape[0]
                units = self.lstm_units
                
                kernel = forward_weights[0]
                kernel_i = kernel[:, :units]
                kernel_f = kernel[:, units:2*units]
                kernel_c = kernel[:, 2*units:3*units]
                kernel_o = kernel[:, 3*units:4*units]
                
                recurrent_kernel = forward_weights[1]
                recurrent_kernel_i = recurrent_kernel[:, :units]
                recurrent_kernel_f = recurrent_kernel[:, units:2*units]
                recurrent_kernel_c = recurrent_kernel[:, 2*units:3*units]
                recurrent_kernel_o = recurrent_kernel[:, 3*units:4*units]
                
                bias = forward_weights[2]
                bias_i = bias[:units]
                bias_f = bias[units:2*units]
                bias_c = bias[2*units:3*units]
                bias_o = bias[3*units:4*units]
                
                forward_lstm_weights = [
                    [kernel_i, recurrent_kernel_i, bias_i],
                    [kernel_f, recurrent_kernel_f, bias_f],
                    [kernel_c, recurrent_kernel_c, bias_c],
                    [kernel_o, recurrent_kernel_o, bias_o]
                ]
                
                backward_weights = backward_layer.get_weights()
                
                kernel = backward_weights[0]
                kernel_i = kernel[:, :units]
                kernel_f = kernel[:, units:2*units]
                kernel_c = kernel[:, 2*units:3*units]
                kernel_o = kernel[:, 3*units:4*units]
                
                recurrent_kernel = backward_weights[1]
                recurrent_kernel_i = recurrent_kernel[:, :units]
                recurrent_kernel_f = recurrent_kernel[:, units:2*units]
                recurrent_kernel_c = recurrent_kernel[:, 2*units:3*units]
                recurrent_kernel_o = recurrent_kernel[:, 3*units:4*units]
                
                bias = backward_weights[2]
                bias_i = bias[:units]
                bias_f = bias[units:2*units]
                bias_c = bias[2*units:3*units]
                bias_o = bias[3*units:4*units]
                
                backward_lstm_weights = [
                    [kernel_i, recurrent_kernel_i, bias_i],
                    [kernel_f, recurrent_kernel_f, bias_f],
                    [kernel_c, recurrent_kernel_c, bias_c],
                    [kernel_o, recurrent_kernel_o, bias_o]
                ]
                
                self.layers[layer_index][1].forward_lstm = LSTM(self.lstm_units, return_sequences=True, weights=forward_lstm_weights)
                self.layers[layer_index][1].backward_lstm = LSTM(self.lstm_units, return_sequences=True, weights=backward_lstm_weights)
                
            else:
                lstm_layer = keras_model.get_layer(index=keras_layer_index)
                lstm_weights = lstm_layer.get_weights()
                
                input_dim = lstm_weights[0].shape[0]
                units = self.lstm_units
                
                kernel = lstm_weights[0]
                kernel_i = kernel[:, :units]
                kernel_f = kernel[:, units:2*units]
                kernel_c = kernel[:, 2*units:3*units]
                kernel_o = kernel[:, 3*units:4*units]
                
                recurrent_kernel = lstm_weights[1]
                recurrent_kernel_i = recurrent_kernel[:, :units]
                recurrent_kernel_f = recurrent_kernel[:, units:2*units]
                recurrent_kernel_c = recurrent_kernel[:, 2*units:3*units]
                recurrent_kernel_o = recurrent_kernel[:, 3*units:4*units]
                
                bias = lstm_weights[2]
                bias_i = bias[:units]
                bias_f = bias[units:2*units]
                bias_c = bias[2*units:3*units]
                bias_o = bias[3*units:4*units]
                
                lstm_weights_split = [
                    [kernel_i, recurrent_kernel_i, bias_i],
                    [kernel_f, recurrent_kernel_f, bias_f],
                    [kernel_c, recurrent_kernel_c, bias_c],
                    [kernel_o, recurrent_kernel_o, bias_o]
                ]
                
                self.layers[layer_index][1].W_i, self.layers[layer_index][1].U_i, self.layers[layer_index][1].b_i = lstm_weights_split[0]
                self.layers[layer_index][1].W_f, self.layers[layer_index][1].U_f, self.layers[layer_index][1].b_f = lstm_weights_split[1]
                self.layers[layer_index][1].W_c, self.layers[layer_index][1].U_c, self.layers[layer_index][1].b_c = lstm_weights_split[2]
                self.layers[layer_index][1].W_o, self.layers[layer_index][1].U_o, self.layers[layer_index][1].b_o = lstm_weights_split[3]
            
            layer_index += 1
            keras_layer_index += 1
            
            layer_index += 1
            keras_layer_index += 1
        
        dense_layer = keras_model.get_layer(index=-1)
        dense_weights = dense_layer.get_weights()
        self.layers[-1][1].weights = dense_weights[0]
        self.layers[-1][1].bias = dense_weights[1]

def batch_predict(model, vectorizer, X, batch_size=32):
    n_samples = len(X)
    n_batches = int(np.ceil(n_samples / batch_size))
    predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_X = X[start_idx:end_idx]
        
        batch_X_vec = vectorizer(batch_X).numpy()
        
        batch_preds = model.forward(batch_X_vec)
        predictions.append(batch_preds)
    
    return np.vstack(predictions)

def load_and_preprocess_test_data(test_csv_path):
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