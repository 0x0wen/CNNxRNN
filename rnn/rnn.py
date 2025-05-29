import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

def tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if x is None:
        raise ValueError("Input 'x' to softmax_np cannot be None.")
    max_x = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def relu_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

ACTIVATION_FUNCTIONS_NP: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'tanh': tanh_np,
    'softmax': softmax_np,
    'relu': relu_np,
    'sigmoid': sigmoid_np,
    'linear': lambda x: x
}

def dtanh_np(activated_output: np.ndarray) -> np.ndarray:
    return 1 - activated_output**2

def drelu_np(activated_output: np.ndarray) -> np.ndarray:
    return (activated_output > 0).astype(activated_output.dtype)

def dsigmoid_np(activated_output: np.ndarray) -> np.ndarray:
    return activated_output * (1 - activated_output)

ACTIVATION_DERIVATIVES_NP: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'tanh': dtanh_np,
    'relu': drelu_np,
    'sigmoid': dsigmoid_np,
    'linear': lambda x: np.ones_like(x)
}

def cross_entropy_loss_np(y_pred_proba: np.ndarray, y_true_int: np.ndarray, num_classes: int) -> float:
    batch_size = y_pred_proba.shape[0]
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-12, 1. - 1e-12)
    log_likelihood = -np.log(y_pred_proba_clipped[range(batch_size), y_true_int])
    loss = np.sum(log_likelihood) / batch_size
    return loss

def d_cross_entropy_softmax_np(y_pred_proba: np.ndarray, y_true_int: np.ndarray, num_classes: int) -> np.ndarray:
    batch_size = y_pred_proba.shape[0]
    y_true_one_hot = np.zeros((batch_size, num_classes))
    y_true_one_hot[np.arange(batch_size), y_true_int] = 1
    dL_dLogits = y_pred_proba - y_true_one_hot
    return dL_dLogits / batch_size
class EmbeddingLayerNP:
    def __init__(self, weights: np.ndarray):
        self.weights: np.ndarray = weights
        self.vocab_size, self.embedding_dim = self.weights.shape
        self.input_tokens_cache: Optional[np.ndarray] = None
        self.gradients: Dict[str, np.ndarray] = {}

    def forward(self, x_tokens: np.ndarray) -> np.ndarray:
        if x_tokens is None:
            raise ValueError("Input x_tokens to EmbeddingLayerNP.forward cannot be None.")
        if x_tokens.ndim != 2:
            raise ValueError(f"Input x_tokens must be 2D (batch_size, sequence_length), got {x_tokens.ndim}D")
        self.input_tokens_cache = x_tokens 
        batch_size, sequence_length = x_tokens.shape
        output_array = np.zeros((batch_size, sequence_length, self.embedding_dim))
        for i in range(batch_size):
            for j in range(sequence_length):
                token_idx = x_tokens[i, j]
                if 0 <= token_idx < self.vocab_size:
                    output_array[i, j, :] = self.weights[token_idx]
        return output_array

    def backward(self, dL_dOutput: np.ndarray) -> None: 
        if self.input_tokens_cache is None:
            raise ValueError("Forward pass must be called before backward pass for EmbeddingLayerNP.")
        
        self.gradients['weights'] = np.zeros_like(self.weights)
        flat_tokens = self.input_tokens_cache.flatten()
        flat_dL_dOutput = dL_dOutput.reshape(-1, self.embedding_dim)
        
        valid_indices_mask = (flat_tokens >= 0) & (flat_tokens < self.vocab_size)
        valid_flat_tokens = flat_tokens[valid_indices_mask]
        valid_flat_dL_dOutput = flat_dL_dOutput[valid_indices_mask]
        
        np.add.at(self.gradients['weights'], valid_flat_tokens, valid_flat_dL_dOutput)

    def update_weights(self, learning_rate: float):
        if 'weights' in self.gradients:
            self.weights -= learning_rate * self.gradients['weights']
            self.gradients = {} 

class SimpleRNNLayerNP:
    def __init__(self, units: int, activation_fn_str: str = 'tanh', return_sequences: bool = False, go_backwards: bool = False):
        self.units: int = units
        self.activation_fn_str: str = activation_fn_str
        self.activation_fn: Callable[[np.ndarray], np.ndarray] = ACTIVATION_FUNCTIONS_NP.get(activation_fn_str, tanh_np)
        self.activation_derivative_fn: Optional[Callable[[np.ndarray], np.ndarray]] = ACTIVATION_DERIVATIVES_NP.get(activation_fn_str)
        if not self.activation_derivative_fn:
            raise ValueError(f"Derivative for activation {activation_fn_str} not found.")

        self.return_sequences: bool = return_sequences
        self.go_backwards: bool = go_backwards
        
        self.kernel: Optional[np.ndarray] = None
        self.recurrent_kernel: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        
        self.x_seq_cache: Optional[np.ndarray] = None
        self.h_states_cache: Optional[np.ndarray] = None 
        self.pre_activations_cache: Optional[np.ndarray] = None
        self.gradients: Dict[str, np.ndarray] = {}
        # print(f"Iniialized SimpleRNNLayerNP: units={units}, activation={activation_fn_str}, return_sequences={return_sequences}, go_backwards={go_backwards}")

    def load_weights(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray):
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias
        # print(f"SimpleRNNLayerNP weights loaded: kernel_shape={kernel.shape}, recurrent_kernel_shape={recurrent_kernel.shape}, bias_shape={bias.shape}")

    def forward(self, x_seq: np.ndarray, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("Weights not loaded for SimpleRNNLayerNP. Call load_weights() first.")
        if x_seq is None:
            raise ValueError("Input x_seq to SimpleRNNLayerNP.forward cannot be None.")


        batch_size, timesteps, input_features = x_seq.shape
        if self.kernel.shape[0] != input_features:
            raise ValueError(f"Input feature size ({input_features}) does not match SimpleRNN kernel input dim ({self.kernel.shape[0]}).")
        if self.recurrent_kernel.shape[0] != self.units or self.recurrent_kernel.shape[1] != self.units:
             raise ValueError(f"SimpleRNN recurrent kernel shape mismatch. Expected ({self.units}, {self.units}), got {self.recurrent_kernel.shape}.")


        self.x_seq_cache = x_seq.copy()
        self.h_states_cache = np.zeros((batch_size, timesteps + 1, self.units))
        self.pre_activations_cache = np.zeros((batch_size, timesteps, self.units))
        
        h_t_prev = np.zeros((batch_size, self.units)) if initial_state is None else initial_state.copy()
        self.h_states_cache[:, 0, :] = h_t_prev
        
        all_output_hidden_states = np.zeros((batch_size, timesteps, self.units))
        
        current_x_for_processing = x_seq
        if self.go_backwards:
            current_x_for_processing = x_seq[:, ::-1, :] 

        for t in range(timesteps): 
            x_t = current_x_for_processing[:, t, :]
            
            pre_activation = np.dot(x_t, self.kernel) + np.dot(h_t_prev, self.recurrent_kernel) + self.bias
            self.pre_activations_cache[:, t, :] = pre_activation
            
            h_t = self.activation_fn(pre_activation)
            all_output_hidden_states[:, t, :] = h_t
            
            h_t_prev = h_t 
            self.h_states_cache[:, t + 1, :] = h_t

        if self.go_backwards and self.return_sequences:
            all_output_hidden_states = all_output_hidden_states[:, ::-1, :]
        
        if self.return_sequences:
            return all_output_hidden_states
        else:
            return h_t_prev

    def backward(self, dL_dOutput: np.ndarray) -> np.ndarray:
        if self.x_seq_cache is None or self.h_states_cache is None or \
           self.pre_activations_cache is None or self.kernel is None or \
           self.recurrent_kernel is None or self.bias is None or self.activation_derivative_fn is None:
            raise ValueError("RNN Forward pass, weights load, or derivative setup incomplete for backward pass.")

        batch_size, timesteps, _ = self.x_seq_cache.shape
        
        self.gradients['kernel'] = np.zeros_like(self.kernel)
        self.gradients['recurrent_kernel'] = np.zeros_like(self.recurrent_kernel)
        self.gradients['bias'] = np.zeros_like(self.bias)
        dL_dX_seq_chronological = np.zeros_like(self.x_seq_cache)
        
        dL_dh_next_t = np.zeros((batch_size, self.units)) 

        for t_proc in reversed(range(timesteps)):
            t_chrono = (timesteps - 1 - t_proc) if self.go_backwards else t_proc
            dL_dh_t_from_layer_output = np.zeros_like(dL_dh_next_t)
            if self.return_sequences:
                dL_dh_t_from_layer_output = dL_dOutput[:, t_chrono, :]
            elif t_proc == (timesteps - 1): 
                dL_dh_t_from_layer_output = dL_dOutput
            
            dL_dh_proc_t = dL_dh_t_from_layer_output + dL_dh_next_t
            activated_h_proc_t = self.h_states_cache[:, t_proc + 1, :]
            dL_dPreActivation_proc_t = dL_dh_proc_t * self.activation_derivative_fn(activated_h_proc_t)
            
            x_input_to_cell_at_t_proc = (self.x_seq_cache[:, ::-1, :] if self.go_backwards else self.x_seq_cache)[:, t_proc, :]
            h_prev_input_to_cell_at_t_proc = self.h_states_cache[:, t_proc, :]

            self.gradients['bias'] += np.sum(dL_dPreActivation_proc_t, axis=0)
            self.gradients['kernel'] += np.dot(x_input_to_cell_at_t_proc.T, dL_dPreActivation_proc_t)
            self.gradients['recurrent_kernel'] += np.dot(h_prev_input_to_cell_at_t_proc.T, dL_dPreActivation_proc_t)
            
            dL_dh_next_t = np.dot(dL_dPreActivation_proc_t, self.recurrent_kernel.T)
            dL_dX_seq_chronological[:, t_chrono, :] = np.dot(dL_dPreActivation_proc_t, self.kernel.T)
            
        return dL_dX_seq_chronological

    def update_weights(self, learning_rate: float):
        if 'kernel' in self.gradients:
            self.kernel -= learning_rate * self.gradients['kernel']
            self.recurrent_kernel -= learning_rate * self.gradients['recurrent_kernel']
            self.bias -= learning_rate * self.gradients['bias']
            self.gradients = {}

class BidirectionalWrapperNP:
    def __init__(self, forward_layer: SimpleRNNLayerNP, backward_layer: SimpleRNNLayerNP, return_sequences: bool):
        if not backward_layer.go_backwards:
            raise ValueError("Backward layer in BidirectionalWrapperNP must have go_backwards=True.")
        if forward_layer.return_sequences != backward_layer.return_sequences:
            raise ValueError("Forward and backward layers must have the same return_sequences setting for BidirectionalWrapperNP.")
        if forward_layer.return_sequences != return_sequences:
             print(f"Warning: BidirectionalWrapperNP.return_sequences ({return_sequences}) differs from internal layers ({forward_layer.return_sequences}). Using internal layer's setting for processing logic, wrapper's for final shape determination if different.")

        self.forward_layer: SimpleRNNLayerNP = forward_layer
        self.backward_layer: SimpleRNNLayerNP = backward_layer
        self.return_sequences = return_sequences 
        # print(f"Initialized BidirectionalWrapperNP: return_sequences={self.return_sequences}")
        
    def forward(self, x_seq: np.ndarray, initial_state_fw: Optional[np.ndarray] = None, initial_state_bw: Optional[np.ndarray] = None) -> np.ndarray:
        if x_seq is None:
            raise ValueError("Input x_seq to BidirectionalWrapperNP.forward cannot be None.")
        output_fw = self.forward_layer.forward(x_seq, initial_state=initial_state_fw)
        output_bw = self.backward_layer.forward(x_seq, initial_state=initial_state_bw)
        
        if self.return_sequences: 
            return np.concatenate([output_fw, output_bw], axis=-1)
        else:
            return np.concatenate([output_fw, output_bw], axis=-1)

    def backward(self, dL_dOutput_concat: np.ndarray) -> np.ndarray:
        num_units_fw = self.forward_layer.units
        
        if self.return_sequences: 
            dL_dOutput_fw = dL_dOutput_concat[:, :, :num_units_fw]
            dL_dOutput_bw = dL_dOutput_concat[:, :, num_units_fw:]
        else: 
            dL_dOutput_fw = dL_dOutput_concat[:, :num_units_fw]
            dL_dOutput_bw = dL_dOutput_concat[:, num_units_fw:]

        dL_dX_fw = self.forward_layer.backward(dL_dOutput_fw)
        dL_dX_bw = self.backward_layer.backward(dL_dOutput_bw)
        
        dL_dInput = dL_dX_fw + dL_dX_bw
        return dL_dInput

    def update_weights(self, learning_rate: float):
        self.forward_layer.update_weights(learning_rate)
        self.backward_layer.update_weights(learning_rate)

class DropoutLayerNP:
    def __init__(self, rate: float):
        self.rate: float = rate
        self.mask: Optional[np.ndarray] = None
        # print(f"Initialized DropoutLayerNP: rate={rate}")

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if x is None:
            print("Warning: Input 'x' to DropoutLayerNP.forward is None.")
            return None
            
        if training:
            if self.rate < 0 or self.rate > 1:
                raise ValueError("Dropout rate must be between 0 and 1.")
            if self.rate == 1.0: 
                self.mask = np.zeros_like(x, dtype=bool)
                return np.zeros_like(x)
            
            self.mask = (np.random.rand(*x.shape) > self.rate)
            return (x * self.mask) / (1 - self.rate)
        else:
            self.mask = None 
            return x

    def backward(self, dL_dOutput: np.ndarray, training: bool = False) -> np.ndarray:
        if dL_dOutput is None:
            print("Warning: dL_dOutput to DropoutLayerNP.backward is None.")
            return None

        if training:
            if self.mask is None: 
                print("Warning: Dropout backward called in training mode but mask is None. Was forward(training=True) called?")
                return dL_dOutput 
            if self.rate == 1.0:
                return np.zeros_like(dL_dOutput)
            return (dL_dOutput * self.mask) / (1 - self.rate)
        else:
            return dL_dOutput

    def update_weights(self, learning_rate: float): 
        pass

class DenseLayerNP:
    def __init__(self, units: int, activation_fn_str: Optional[str] = None):
        self.units: int = units
        self.activation_fn_str: Optional[str] = activation_fn_str
        self.activation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.activation_derivative_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

        if activation_fn_str and activation_fn_str != 'linear':
            self.activation_fn = ACTIVATION_FUNCTIONS_NP.get(activation_fn_str)
            if not self.activation_fn:
                raise ValueError(f"Unsupported activation function for DenseLayerNP: {activation_fn_str}")
            if activation_fn_str != 'softmax': 
                 self.activation_derivative_fn = ACTIVATION_DERIVATIVES_NP.get(activation_fn_str)
                 if not self.activation_derivative_fn:
                     raise ValueError(f"Derivative for {activation_fn_str} not found for DenseLayerNP.")
        
        self.kernel: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.input_cache: Optional[np.ndarray] = None
        self.pre_activation_cache: Optional[np.ndarray] = None
        self.activated_output_cache: Optional[np.ndarray] = None
        self.gradients: Dict[str, np.ndarray] = {}

    def load_weights(self, kernel: np.ndarray, bias: np.ndarray):
        self.kernel = kernel
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.bias is None:
            raise ValueError("Weights (kernel or bias) not loaded for DenseLayerNP. Call load_weights() first.")
        if x is None:
            raise ValueError("Input 'x' to DenseLayerNP.forward cannot be None. Output of previous layer might be None.")

        self.input_cache = x
        
        try:
            self.pre_activation_cache = np.dot(x, self.kernel) + self.bias
        except TypeError as e:
            print(f"TypeError during np.dot in DenseLayerNP: {e}")
            print(f"  x shape/type: {x.shape if isinstance(x, np.ndarray) else type(x)}")
            print(f"  kernel shape/type: {self.kernel.shape if isinstance(self.kernel, np.ndarray) else type(self.kernel)}")
            print(f"  bias shape/type: {self.bias.shape if isinstance(self.bias, np.ndarray) else type(self.bias)}")
            raise 

        if self.pre_activation_cache is None:
             raise ValueError("self.pre_activation_cache is None in DenseLayerNP after dot product and bias addition. This should not happen if inputs are valid.")

        if self.activation_fn:
            self.activated_output_cache = self.activation_fn(self.pre_activation_cache)
        else:
            self.activated_output_cache = self.pre_activation_cache
        
        if self.activated_output_cache is None:
            raise ValueError("self.activated_output_cache is None in DenseLayerNP after activation. This indicates an issue with the activation function or its input.")
            
        return self.activated_output_cache

    def backward(self, dL_dActivatedOutput: np.ndarray) -> np.ndarray:
        if dL_dActivatedOutput is None:
            raise ValueError("dL_dActivatedOutput to DenseLayerNP.backward cannot be None.")
        if self.input_cache is None or self.pre_activation_cache is None or \
           self.activated_output_cache is None or self.kernel is None or self.bias is None:
            raise ValueError("DenseLayerNP Forward pass, weights load, or cache incomplete for backward pass.")

        dL_dPreActivation: np.ndarray
        if self.activation_fn_str == 'softmax':
            dL_dPreActivation = dL_dActivatedOutput
        elif self.activation_fn_str and self.activation_fn_str != 'linear':
            if not self.activation_derivative_fn:
                 raise ValueError(f"Derivative function not set for {self.activation_fn_str} in DenseLayerNP for backward pass.")
            dL_dPreActivation = dL_dActivatedOutput * self.activation_derivative_fn(self.activated_output_cache)
        else: 
            dL_dPreActivation = dL_dActivatedOutput
        
        if self.input_cache.ndim == 3: # (batch, timesteps, features_in)
            self.gradients['kernel'] = np.einsum('bti,btj->ij', self.input_cache, dL_dPreActivation)
            self.gradients['bias'] = np.sum(dL_dPreActivation, axis=(0, 1))
        else: # (batch, features_in)
            self.gradients['kernel'] = np.dot(self.input_cache.T, dL_dPreActivation)
            self.gradients['bias'] = np.sum(dL_dPreActivation, axis=0)

        dL_dInput = np.dot(dL_dPreActivation, self.kernel.T)
        return dL_dInput

    def update_weights(self, learning_rate: float):
        if 'kernel' in self.gradients and 'bias' in self.gradients:
            self.kernel -= learning_rate * self.gradients['kernel']
            self.bias -= learning_rate * self.gradients['bias']
            self.gradients = {}
class SequentialFromScratch:
    def __init__(self, layers: List[Any]):
        self.layers: List[Any] = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        current_output = x
        for i, layer in enumerate(self.layers):
            if current_output is None :
                 if not (i == 0 and isinstance(layer, EmbeddingLayerNP)):
                    raise ValueError(f"Input to layer {i} ({type(layer).__name__}) is None. Output of previous layer was None.")


            if isinstance(layer, DropoutLayerNP):
                current_output = layer.forward(current_output, training=training)
            elif hasattr(layer, 'forward'):
                current_output = layer.forward(current_output)
            else:
                raise TypeError(f"Layer {type(layer).__name__} does not have a 'forward' method.")
        
        if current_output is None:
            raise ValueError("Final output of SequentialFromScratch is None. Check intermediate layers.")
        return current_output

    def backward(self, dL_dModelOutput: np.ndarray, training: bool = False) -> None:
        current_dL_dLayerInput = dL_dModelOutput
        for layer in reversed(self.layers):
            if current_dL_dLayerInput is None:
                print(f"Warning: Gradient input (dL_dLayerInput) to layer {type(layer).__name__}.backward is None. Stopping backprop for this path.")
                break

            if isinstance(layer, DropoutLayerNP):
                current_dL_dLayerInput = layer.backward(current_dL_dLayerInput, training=training)
            elif isinstance(layer, EmbeddingLayerNP):
                layer.backward(current_dL_dLayerInput) 
                current_dL_dLayerInput = None
            elif hasattr(layer, 'backward'):
                current_dL_dLayerInput = layer.backward(current_dL_dLayerInput)
            else:
                print(f"Warning: Layer {type(layer).__name__} does not have a 'backward' method or is not handled in SequentialFromScratch.backward.")


    def update_weights(self, learning_rate: float):
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(learning_rate)