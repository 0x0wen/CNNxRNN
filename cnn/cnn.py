import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class Layer:
    def __init__(self):
        self.input_data_cache = None
        self.output_data_cache = None
        self.weights = None
        self.bias = None
        self.name = self.__class__.__name__

    def forward(self, input_data):
        raise NotImplementedError("Metode forward harus di-override oleh subclass.")

    def load_params(self, weights=None, bias=None):
        if weights is not None:
            self.weights = weights.astype(np.float32)
        if bias is not None:
            self.bias = bias.astype(np.float32)
    
    def __str__(self):
        return f"Layer: {self.name}"

class Conv2D(Layer):
    def __init__(self, num_filters, filter_size, input_shape_depth=None, stride=1, padding='valid', name=None):
        super().__init__()
        self.num_filters = num_filters
        self.filter_height, self.filter_width = filter_size
        self.stride = stride
        self.padding_mode = padding
        self.input_depth = input_shape_depth
        if name: self.name = name

    def _im2col(self, input_data, filter_h, filter_w, stride):
        N, C, H, W = input_data.shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1

        img = input_data
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col

    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        if self.input_data_cache.ndim != 4:
            raise ValueError(f"Input data untuk {self.name} (Conv2D) harus 4D (batch, depth, height, width). Diterima shape: {self.input_data_cache.shape}")

        batch_size, current_input_depth, input_height, input_width = self.input_data_cache.shape

        if self.input_depth is None:
            self.input_depth = current_input_depth
        elif self.input_depth != current_input_depth:
            raise ValueError(f"Input depth ({current_input_depth}) tidak sesuai dengan input_depth layer {self.name} ({self.input_depth}) yang diharapkan.")

        if self.weights is None or self.bias is None:
            raise RuntimeError(f"Bobot dan bias untuk layer {self.name} (Conv2D) belum di-load.")

        expected_weights_shape = (self.num_filters, self.input_depth, self.filter_height, self.filter_width)
        if self.weights.shape != expected_weights_shape:
            raise ValueError(f"Dimensi bobot {self.name} (Conv2D) tidak sesuai. Diharapkan: {expected_weights_shape}, Didapat: {self.weights.shape}")
        if self.bias.shape != (self.num_filters,):
            raise ValueError(f"Dimensi bias {self.name} (Conv2D) tidak sesuai. Diharapkan: {(self.num_filters,)}, Didapat: {self.bias.shape}")

        if self.padding_mode == 'same':
            pad_h_total = max((input_height - 1) * self.stride + self.filter_height - input_height, 0)
            pad_w_total = max((input_width - 1) * self.stride + self.filter_width - input_width, 0)
            pad_h_before = pad_h_total // 2
            pad_h_after = pad_h_total - pad_h_before
            pad_w_before = pad_w_total // 2
            pad_w_after = pad_w_total - pad_w_before
            input_padded = np.pad(self.input_data_cache,
                                  ((0, 0), (0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                                  mode='constant', constant_values=0)
        elif self.padding_mode == 'valid':
            input_padded = self.input_data_cache
        else:
            raise ValueError(f"Padding tidak valid untuk {self.name}. Pilih 'same' atau 'valid'.")

        _, _, padded_height, padded_width = input_padded.shape
        output_height = (padded_height - self.filter_height) // self.stride + 1
        output_width = (padded_width - self.filter_width) // self.stride + 1

        col = self._im2col(input_padded, self.filter_height, self.filter_width, self.stride)
        col_W = self.weights.reshape(self.num_filters, -1).T

        out = np.dot(col, col_W) + self.bias
        out = out.reshape(batch_size, output_height, output_width, -1).transpose(0, 3, 1, 2)

        self.output_data_cache = out
        return self.output_data_cache

class ReLU(Layer):
    def __init__(self, name=None):
        super().__init__()
        if name: self.name = name

    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        self.output_data_cache = relu(self.input_data_cache)
        return self.output_data_cache

class PoolingBase(Layer): 
    def __init__(self, pool_size=(2, 2), stride=None, padding='valid', name=None):
        super().__init__()
        self.pool_height, self.pool_width = pool_size
        self.stride_val = stride if stride is not None else self.pool_height 
        self.padding_mode = padding
        if name: self.name = name

    def _calculate_padding_and_output_dims(self, input_height, input_width):
        if self.padding_mode == 'valid':
            output_height = (input_height - self.pool_height) // self.stride_val + 1
            output_width = (input_width - self.pool_width) // self.stride_val + 1
            pad_h_before, pad_h_after, pad_w_before, pad_w_after = 0, 0, 0, 0
        elif self.padding_mode == 'same':
            output_height = int(np.ceil(float(input_height) / float(self.stride_val)))
            output_width = int(np.ceil(float(input_width) / float(self.stride_val)))
            pad_h_total = max(0, (output_height - 1) * self.stride_val + self.pool_height - input_height)
            pad_w_total = max(0, (output_width - 1) * self.stride_val + self.pool_width - input_width)
            pad_h_before = pad_h_total // 2
            pad_h_after = pad_h_total - pad_h_before
            pad_w_before = pad_w_total // 2
            pad_w_after = pad_w_total - pad_w_before
        else:
            raise ValueError(f"Padding tidak valid untuk {self.name}. Pilih 'same' atau 'valid'.")
        return output_height, output_width, pad_h_before, pad_h_after, pad_w_before, pad_w_after

class MaxPooling2D(PoolingBase):
    def _pool2d(self, input_data):
        N, C, H, W = input_data.shape
        out_h = (H - self.pool_height) // self.stride_val + 1
        out_w = (W - self.pool_width) // self.stride_val + 1
        
        stride_h = self.stride_val
        stride_w = self.stride_val
        
        windows = np.lib.stride_tricks.as_strided(
            input_data,
            shape=(N, C, out_h, out_w, self.pool_height, self.pool_width),
            strides=(input_data.strides[0], input_data.strides[1],
                    stride_h * input_data.strides[2], stride_w * input_data.strides[3],
                    input_data.strides[2], input_data.strides[3])
        )
        
        return np.max(windows, axis=(4, 5))

    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        if self.input_data_cache.ndim != 4:
            raise ValueError(f"Input data untuk {self.name} (MaxPooling2D) harus 4D. Diterima: {self.input_data_cache.shape}")

        batch_size, input_depth, input_height, input_width = self.input_data_cache.shape
        
        output_height, output_width, pad_h_before, pad_h_after, pad_w_before, pad_w_after = \
            self._calculate_padding_and_output_dims(input_height, input_width)

        if self.padding_mode == 'same' and (pad_h_before > 0 or pad_w_before > 0 or pad_h_after > 0 or pad_w_after > 0):
            input_padded = np.pad(self.input_data_cache,
                                  ((0,0), (0,0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                                  mode='constant', constant_values=-np.inf)
        else:
            input_padded = self.input_data_cache
            
        self.output_data_cache = self._pool2d(input_padded)
        return self.output_data_cache

class AveragePooling2D(PoolingBase):
    def _pool2d(self, input_data):
        N, C, H, W = input_data.shape
        out_h = (H - self.pool_height) // self.stride_val + 1
        out_w = (W - self.pool_width) // self.stride_val + 1
        
        stride_h = self.stride_val
        stride_w = self.stride_val
        
        windows = np.lib.stride_tricks.as_strided(
            input_data,
            shape=(N, C, out_h, out_w, self.pool_height, self.pool_width),
            strides=(input_data.strides[0], input_data.strides[1],
                    stride_h * input_data.strides[2], stride_w * input_data.strides[3],
                    input_data.strides[2], input_data.strides[3])
        )
        
        return np.mean(windows, axis=(4, 5))

    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        if self.input_data_cache.ndim != 4:
            raise ValueError(f"Input data untuk {self.name} (AveragePooling2D) harus 4D. Diterima: {self.input_data_cache.shape}")
        
        batch_size, input_depth, input_height, input_width = self.input_data_cache.shape

        output_height, output_width, pad_h_before, pad_h_after, pad_w_before, pad_w_after = \
            self._calculate_padding_and_output_dims(input_height, input_width)
        
        if self.padding_mode == 'same' and (pad_h_before > 0 or pad_w_before > 0 or pad_h_after > 0 or pad_w_after > 0):
            input_padded = np.pad(self.input_data_cache,
                                  ((0,0), (0,0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                                  mode='constant', constant_values=0)
        else:
            input_padded = self.input_data_cache

        self.output_data_cache = self._pool2d(input_padded)
        return self.output_data_cache

class Flatten(Layer):
    def __init__(self, name=None):
        super().__init__()
        self.original_input_shape_cache = None
        if name: self.name = name

    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        
        if self.input_data_cache.ndim != 4:
            if self.input_data_cache.ndim == 2:
                self.original_input_shape_cache = self.input_data_cache.shape
                self.output_data_cache = self.input_data_cache
                return self.output_data_cache
            raise ValueError(f"Input data untuk {self.name} (Flatten) dari conv/pool diharapkan 4D. Diterima: {self.input_data_cache.shape}")
        
        self.original_input_shape_cache = self.input_data_cache.shape 
        batch_size = self.input_data_cache.shape[0]
        
        input_transposed = self.input_data_cache.transpose(0, 2, 3, 1) 
        self.output_data_cache = input_transposed.reshape(batch_size, -1)
        
        return self.output_data_cache

class Dense(Layer):
    def __init__(self, output_size, input_size=None, name=None):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        if name: self.name = name

    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        if self.input_data_cache.ndim == 1: 
            input_data_batched = self.input_data_cache.reshape(1, -1)
        elif self.input_data_cache.ndim == 2:
            input_data_batched = self.input_data_cache
        else:
            raise ValueError(f"Input data untuk {self.name} (Dense) harus 1D atau 2D. Diterima: {self.input_data_cache.shape}")

        current_input_size = input_data_batched.shape[1]
        if self.input_size is None: 
            self.input_size = current_input_size
        elif self.input_size != current_input_size:
            raise ValueError(f"Input size ({current_input_size}) tidak sesuai dengan input_size layer {self.name} ({self.input_size}) yang diharapkan.")

        if self.weights is None or self.bias is None:
            raise RuntimeError(f"Bobot dan bias untuk layer {self.name} (Dense) belum di-load.")

        expected_weights_shape = (self.input_size, self.output_size)
        if self.weights.shape != expected_weights_shape:
            raise ValueError(f"Dimensi bobot {self.name} (Dense) tidak sesuai. Diharapkan: {expected_weights_shape}, Didapat: {self.weights.shape}")
        if self.bias.shape != (self.output_size,):
             raise ValueError(f"Dimensi bias {self.name} (Dense) tidak sesuai. Diharapkan: {(self.output_size,)}, Didapat: {self.bias.shape}")

        z = np.dot(input_data_batched, self.weights) + self.bias
        self.output_data_cache = z
        
        if self.input_data_cache.ndim == 1 and self.output_data_cache.shape[0] == 1:
            return self.output_data_cache.squeeze(axis=0)
        return self.output_data_cache

class SoftmaxActivation(Layer):
    def __init__(self, name=None):
        super().__init__()
        if name: self.name = name
    
    def forward(self, input_data):
        self.input_data_cache = input_data.astype(np.float32)
        self.output_data_cache = softmax(self.input_data_cache, axis=-1)
        return self.output_data_cache

class CNNModel:
    def __init__(self, layers):
        self.layers = layers
        self.keras_layer_name_map = {} 

    def forward(self, input_data):
        current_output = input_data.astype(np.float32)
        print(f"Initial input shape: {current_output.shape}")
        for i, layer in enumerate(self.layers):
            prev_output_shape = current_output.shape
            try:
                current_output = layer.forward(current_output)
            except Exception as e:
                print(f"ERROR saat forward propagation di layer {i+1:02d} ({layer.name}): {e}")
                print(f"  Input shape ke layer {layer.name}: {prev_output_shape}")
                if hasattr(layer, 'weights') and layer.weights is not None:
                    print(f"  Dimensi bobot layer {layer.name}: {layer.weights.shape}")
                if hasattr(layer, 'bias') and layer.bias is not None:
                    print(f"  Dimensi bias layer {layer.name}: {layer.bias.shape}")
                raise 
            print(f"Layer {i+1:02d} ({layer.name}): {str(prev_output_shape):<25} -> {current_output.shape}")
        return current_output

    def load_keras_weights(self, keras_model_instance):
        keras_layers_with_weights = [l for l in keras_model_instance.layers if l.get_weights()]
        
        scratch_layer_idx = 0
        keras_layer_with_weights_idx = 0
    
        print("\nMemulai pemetaan bobot Keras ke model scratch...")
        while scratch_layer_idx < len(self.layers) and keras_layer_with_weights_idx < len(keras_layers_with_weights):
            current_scratch_layer = self.layers[scratch_layer_idx]
            
            if isinstance(current_scratch_layer, (Conv2D, Dense)):
                keras_layer = keras_layers_with_weights[keras_layer_with_weights_idx]
                self.keras_layer_name_map[current_scratch_layer.name] = keras_layer.name

                keras_weights_list = keras_layer.get_weights()
                weights = keras_weights_list[0]
                bias = keras_weights_list[1]

                if isinstance(current_scratch_layer, Conv2D):
                    weights_transposed = weights.transpose(3, 2, 0, 1)
                    current_scratch_layer.load_params(weights_transposed, bias)
                    print(f"  Loaded Conv2D {current_scratch_layer.name} - Keras original weights: {weights.shape}, "
                          f"Transposed: {weights_transposed.shape}, Bias: {bias.shape}")
                    
                    if current_scratch_layer.input_depth is None:
                        print(f"  Warning, input_depth untuk {current_scratch_layer.name} masih None setelah inisialisasi.")

                elif isinstance(current_scratch_layer, Dense):
                    current_scratch_layer.load_params(weights, bias)
                    print(f"  Loaded Dense {current_scratch_layer.name} - Weights: {weights.shape}, Bias: {bias.shape}")

                    if current_scratch_layer.input_size is None:
                         print(f"  Warning, input_size untuk {current_scratch_layer.name} masih None setelah inisialisasi.")
                
                keras_layer_with_weights_idx += 1 
            
            scratch_layer_idx += 1 
            
        if keras_layer_with_weights_idx < len(keras_layers_with_weights):
            remaining_keras_layers = [l.name for l in keras_layers_with_weights[keras_layer_with_weights_idx:]]
            print(f"Peringatan (load_keras_weights): Masih ada Keras layers dengan bobot yang belum dipetakan: {remaining_keras_layers}")
        
        remaining_scratch_layers_needing_weights = [
            l.name for l in self.layers[scratch_layer_idx:] if isinstance(l, (Conv2D, Dense))
        ]
        if remaining_scratch_layers_needing_weights:
            print(f"Peringatan (load_keras_weights): Masih ada scratch layers (Conv2D/Dense) yang belum menerima bobot: {remaining_scratch_layers_needing_weights}")

        print("\nPemetaan bobot Keras selesai.")


    def summary(self):
        print("-" * 80)
        print(f"{'Layer (type)':<40} {'Output Shape':<25} {'Keras Mapped Name (Params)'}")
        print("=" * 80)
        total_params = 0
        
        for layer in self.layers:
            output_shape_str = str(layer.output_data_cache.shape) if layer.output_data_cache is not None else "(Belum di-forward)"
            keras_mapped_name = self.keras_layer_name_map.get(layer.name, "N/A")
            
            params = 0
            params_str = ""
            if layer.weights is not None:
                params += np.prod(layer.weights.shape)
            if layer.bias is not None:
                params += np.prod(layer.bias.shape)
            
            if params > 0:
                params_str = f" ({int(params)} params)"
            elif isinstance(layer, (Conv2D, Dense)): 
                params_str = " (Params N/A)"

            display_name = f"{layer.name} ({layer.__class__.__name__})"
            mapped_info = f"{keras_mapped_name}{params_str}"
            print(f"{display_name:<40} {output_shape_str:<25} {mapped_info}")
            total_params += params
        print("=" * 80)
        print(f"Total params (from scratch model, loaded): {int(total_params)}")
        print("-" * 80)