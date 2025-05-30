import argparse
import os
import json
from typing import Optional
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import f1_score


from train import (
    run_training_pipeline,
    load_and_preprocess_data_for_testing,
    MODEL_DIR,
)

from rnn import (
    SequentialFromScratch,
    SimpleRNNLayerNP,
    BidirectionalWrapperNP,
    DropoutLayerNP,
    DenseLayerNP,
    EmbeddingLayerNP,
    d_cross_entropy_softmax_np
)

MAX_FEATURES = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
NUM_CLASSES = 3

def load_keras_model_and_vectorizer(model_config_name: str):
    if model_config_name == "best_model":
        keras_model_filename = "best_model.keras"
    else:
        keras_model_filename = f"model_{model_config_name}.keras"
    
    keras_model_path_full = os.path.join(MODEL_DIR, keras_model_filename)

    if not os.path.exists(keras_model_path_full):
        print(f"Error: Keras model not found for '{model_config_name}'.")
        print(f"Expected Keras model: {keras_model_path_full}")
        if model_config_name != "best_model":
            print("Please ensure --train has been run for this configuration and the name is correct.")
        else:
            print("Please ensure --train has been run to generate 'best_model.keras'.")
        return None, None
        
    print(f"Loading Keras model from: {keras_model_path_full}")
    try:
        keras_model = load_model(keras_model_path_full, compile=False) 
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None, None

    retrieved_keras_vectorizer = None
    if keras_model.layers and isinstance(keras_model.layers[0], tf.keras.layers.TextVectorization):
        retrieved_keras_vectorizer = keras_model.layers[0]
        print("TextVectorization layer successfully retrieved from the loaded Keras model.")
    else:
        print("Error: First layer of loaded Keras model is not TextVectorization or model has no layers.")
        print("The --test-scratch functionality relies on the Keras model containing the TextVectorization layer.")
        return None, None 

    return keras_model, retrieved_keras_vectorizer


def run_test_scratch(model_config_name: str, inference_batch_size: Optional[int] = None, demo_backward_pass: bool = False):
    print(f"\n--- Running From-Scratch Test for model: {model_config_name} ---")
    if inference_batch_size is not None:
        print(f"--- Using inference batch size for scratch model: {inference_batch_size} ---")

    keras_model, text_vectorizer_keras = load_keras_model_and_vectorizer(model_config_name)
    if keras_model is None or text_vectorizer_keras is None:
        print("Failed to load Keras model or retrieve its vectorizer. Aborting scratch test.")
        return

    loaded_data = load_and_preprocess_data_for_testing(for_scratch_test=True)
    if loaded_data is None or loaded_data[4] is None or loaded_data[5] is None: 
        print("Failed to load test data.")
        return
    test_texts = loaded_data[4] 
    test_labels_raw = loaded_data[5]

    print("Vectorizing test texts using the Keras model's TextVectorization layer...")
    try:
        vectorized_test_data_tf = text_vectorizer_keras(tf.constant(test_texts, dtype=tf.string))
    except Exception as e:
        print(f"Error during TextVectorization of test_texts: {e}")
        return
        
    vectorized_test_data_np = vectorized_test_data_tf.numpy()
    print(f"Shape of vectorized test data (NumPy): {vectorized_test_data_np.shape}")

    num_rnn_layers_cfg: int
    rnn_units_cfg: int
    direction_cfg: str
    
    if model_config_name == "best_model":
        best_model_config_path = os.path.join(MODEL_DIR, "best_model_config.json")
        if not os.path.exists(best_model_config_path):
            print(f"Error: 'best_model_config.json' not found at {best_model_config_path}.")
            print("This file is needed to determine the architecture of 'best_model.keras' for the scratch model.")
            print("Please ensure --train has been run and a best model (with its config) was saved.")
            return
        try:
            with open(best_model_config_path, 'r') as f:
                config_data = json.load(f)
            num_rnn_layers_cfg = int(config_data['num_rnn_layers'])
            rnn_units_cfg = int(config_data['rnn_units'])
            direction_cfg = config_data['rnn_direction']
            if direction_cfg not in ['unidirectional', 'bidirectional']:
                raise ValueError(f"Invalid direction '{direction_cfg}' in '{best_model_config_path}'.")
            print(f"Loaded configuration for 'best_model' from JSON: Layers={num_rnn_layers_cfg}, Units={rnn_units_cfg}, Direction={direction_cfg}")
        except Exception as e:
            print(f"Error loading or parsing '{best_model_config_path}': {e}")
            return
    else:
        parts = model_config_name.split('_')
        try:
            num_rnn_layers_cfg = int(parts[parts.index('layers') + 1])
            rnn_units_cfg = int(parts[parts.index('units') + 1])
            direction_cfg = parts[parts.index('dir') + 1]
            if direction_cfg not in ['unidirectional', 'bidirectional']:
                raise ValueError(f"Invalid direction '{direction_cfg}' in model configuration name.")
        except (ValueError, IndexError) as e:
            print(f"Error parsing model configuration from name '{model_config_name}': {e}")
            return
        
    is_bidirectional_cfg = (direction_cfg == 'bidirectional')

    scratch_layers = []
    current_keras_layer_idx = 0 

    if isinstance(keras_model.layers[current_keras_layer_idx], tf.keras.layers.TextVectorization):
        current_keras_layer_idx += 1
    else:
        raise ValueError("Keras model's first layer is not TextVectorization. Inconsistency.")

    if current_keras_layer_idx < len(keras_model.layers) and \
       isinstance(keras_model.layers[current_keras_layer_idx], tf.keras.layers.Embedding):
        keras_embedding_layer = keras_model.layers[current_keras_layer_idx]
        scratch_embedding_weights = keras_embedding_layer.get_weights()[0].copy()
        scratch_embedding_layer = EmbeddingLayerNP(weights=scratch_embedding_weights)
        scratch_layers.append(scratch_embedding_layer)
        current_keras_layer_idx += 1
    else:
        raise ValueError(f"Could not find Embedding layer at expected position in Keras model. Found: {type(keras_model.layers[current_keras_layer_idx]) if current_keras_layer_idx < len(keras_model.layers) else 'No more layers'}")

    for i in range(num_rnn_layers_cfg):
        if current_keras_layer_idx >= len(keras_model.layers):
            raise ValueError(f"Keras model does not have enough layers to match num_rnn_layers_cfg={num_rnn_layers_cfg}")
            
        keras_rnn_candidate_layer = keras_model.layers[current_keras_layer_idx]
        return_sequences_np = (i < num_rnn_layers_cfg - 1) 
        
        if isinstance(keras_rnn_candidate_layer, tf.keras.layers.Bidirectional):
            if not is_bidirectional_cfg: 
                raise ValueError("Model config mismatch: Keras has Bidirectional RNN, but config suggests Unidirectional.")
            
            fw_rnn_keras = keras_rnn_candidate_layer.forward_layer
            bw_rnn_keras = keras_rnn_candidate_layer.backward_layer

            if not (isinstance(fw_rnn_keras, tf.keras.layers.SimpleRNN) and isinstance(bw_rnn_keras, tf.keras.layers.SimpleRNN)):
                 print(f"Warning: Bidirectional Keras layer contains non-SimpleRNN cells ({type(fw_rnn_keras).__name__}), but scratch test uses SimpleRNN.")

            keras_rnn_w = keras_rnn_candidate_layer.get_weights()
            if len(keras_rnn_w) < 6:
                raise ValueError(f"Bidirectional layer {keras_rnn_candidate_layer.name} does not have enough weights (expected 6, got {len(keras_rnn_w)}).")

            fw_kernel, fw_recurrent_kernel, fw_bias = keras_rnn_w[0].copy(), keras_rnn_w[1].copy(), keras_rnn_w[2].copy()
            bw_kernel, bw_recurrent_kernel, bw_bias = keras_rnn_w[3].copy(), keras_rnn_w[4].copy(), keras_rnn_w[5].copy()

            fw_rnn_np = SimpleRNNLayerNP(units=fw_rnn_keras.units, activation_fn_str=fw_rnn_keras.activation.__name__, return_sequences=return_sequences_np, go_backwards=False)
            fw_rnn_np.load_weights(kernel=fw_kernel, recurrent_kernel=fw_recurrent_kernel, bias=fw_bias)
            
            bw_rnn_np = SimpleRNNLayerNP(units=bw_rnn_keras.units, activation_fn_str=bw_rnn_keras.activation.__name__, return_sequences=return_sequences_np, go_backwards=True)
            bw_rnn_np.load_weights(kernel=bw_kernel, recurrent_kernel=bw_recurrent_kernel, bias=bw_bias)
            
            bi_rnn_np = BidirectionalWrapperNP(forward_layer=fw_rnn_np, backward_layer=bw_rnn_np, return_sequences=return_sequences_np)
            scratch_layers.append(bi_rnn_np)
            current_keras_layer_idx += 1

        elif isinstance(keras_rnn_candidate_layer, (tf.keras.layers.SimpleRNN, tf.keras.layers.GRU, tf.keras.layers.LSTM)): 
            if is_bidirectional_cfg: 
                raise ValueError("Model config mismatch: Keras has Unidirectional RNN, but config suggests Bidirectional.")
            if not isinstance(keras_rnn_candidate_layer, tf.keras.layers.SimpleRNN):
                 print(f"Warning: Unidirectional Keras layer is of type {type(keras_rnn_candidate_layer).__name__}, but scratch test uses SimpleRNN.")


            keras_rnn_w = keras_rnn_candidate_layer.get_weights() 
            if len(keras_rnn_w) < 3:
                 raise ValueError(f"Unidirectional RNN layer {keras_rnn_candidate_layer.name} does not have enough weights (expected 3, got {len(keras_rnn_w)}).")
            
            uni_kernel, uni_recurrent_kernel, uni_bias = keras_rnn_w[0].copy(), keras_rnn_w[1].copy(), keras_rnn_w[2].copy()

            rnn_np = SimpleRNNLayerNP(units=keras_rnn_candidate_layer.units, activation_fn_str=keras_rnn_candidate_layer.activation.__name__, return_sequences=return_sequences_np, go_backwards=False)
            rnn_np.load_weights(kernel=uni_kernel, recurrent_kernel=uni_recurrent_kernel, bias=uni_bias)
            scratch_layers.append(rnn_np)
            current_keras_layer_idx += 1
        else:
            raise ValueError(f"Unexpected Keras RNN layer type encountered: {type(keras_rnn_candidate_layer)} at index {current_keras_layer_idx}.")

    if current_keras_layer_idx < len(keras_model.layers) and \
       isinstance(keras_model.layers[current_keras_layer_idx], tf.keras.layers.Dropout):
        keras_dropout_layer = keras_model.layers[current_keras_layer_idx]
        scratch_dropout_layer = DropoutLayerNP(rate=keras_dropout_layer.rate)
        scratch_layers.append(scratch_dropout_layer)
        current_keras_layer_idx +=1
    
    if current_keras_layer_idx < len(keras_model.layers) and \
       isinstance(keras_model.layers[current_keras_layer_idx], tf.keras.layers.Dense):
        keras_dense_layer = keras_model.layers[current_keras_layer_idx]
    else:
        raise ValueError(f"Expected Dense layer, got {type(keras_model.layers[current_keras_layer_idx]) if current_keras_layer_idx < len(keras_model.layers) else 'No more layers'}")
    
    dense_weights_keras = keras_dense_layer.get_weights()
    if len(dense_weights_keras) < 2:
        raise ValueError(f"Dense layer {keras_dense_layer.name} does not have enough weights (expected 2, got {len(dense_weights_keras)}).")

    dense_activation = keras_dense_layer.activation.__name__ if keras_dense_layer.activation else 'linear'
    if dense_activation != 'softmax':
        print(f"Warning: Keras final Dense layer activation is '{dense_activation}', not 'softmax'.")
        print("The from-scratch DenseLayerNP will use 'softmax' as configured in rnn.py for probability output.")

    dense_kernel, dense_bias = dense_weights_keras[0].copy(), dense_weights_keras[1].copy()
    scratch_dense_layer = DenseLayerNP(units=keras_dense_layer.units, activation_fn_str='softmax') 
    scratch_dense_layer.load_weights(kernel=dense_kernel, bias=dense_bias)
    scratch_layers.append(scratch_dense_layer)

    scratch_model = SequentialFromScratch(layers=scratch_layers)
    
    if demo_backward_pass:
        print("\n--- Running Backward Pass Demonstration ---")
        demo_batch_size = min(5, vectorized_test_data_np.shape[0])
        if demo_batch_size == 0:
            print("Not enough test data for backward pass demo.")
        else:
            demo_input_batch = vectorized_test_data_np[:demo_batch_size]
            demo_labels_batch = test_labels_raw[:demo_batch_size]
            print(f"Using {demo_batch_size} samples for backward pass demo.")
            print("1. Performing forward pass (training=True)...")
            predictions_proba_demo = scratch_model.forward(demo_input_batch, training=True)
            print("2. Calculating initial gradient (dL/dLogits)...")
            dL_dLogits_demo = d_cross_entropy_softmax_np(predictions_proba_demo, demo_labels_batch, NUM_CLASSES)
            print("3. Performing backward pass (training=True)...")
            scratch_model.backward(dL_dLogits_demo, training=True)
            print("Backward pass demo executed. Gradients have been computed in scratch model layers (weights NOT updated).")
            for i, layer in enumerate(scratch_model.layers):
                if hasattr(layer, 'gradients') and layer.gradients:
                    print(f"  Gradients computed for layer {i} ({type(layer).__name__}):")
                    for grad_name, grad_val in layer.gradients.items():
                        print(f"    - {grad_name}: shape {grad_val.shape}, sum {np.sum(grad_val):.4e}")
            print("--- End of Backward Pass Demonstration ---\n")
            
    print("Getting from-scratch model predictions (for inference comparison)...")
    if inference_batch_size is None:
        scratch_predictions_proba = scratch_model.forward(vectorized_test_data_np, training=False)
    else:
        num_samples = vectorized_test_data_np.shape[0]
        all_outputs_scratch = []
        print(f"Running scratch model inference in batches of size {inference_batch_size}...")
        for i_batch in range(0, num_samples, inference_batch_size):
            batch_input_np = vectorized_test_data_np[i_batch:min(i_batch + inference_batch_size, num_samples)]
            batch_output_scratch = scratch_model.forward(batch_input_np, training=False) 
            all_outputs_scratch.append(batch_output_scratch)
        scratch_predictions_proba = np.concatenate(all_outputs_scratch, axis=0)

    print("Getting Keras model predictions (for comparison)...")
    keras_predictions_proba = keras_model.predict(tf.constant(test_texts, dtype=tf.string), batch_size=inference_batch_size if inference_batch_size else 32)

    print(f"Shape of Keras predicted probabilities: {keras_predictions_proba.shape}")
    print(f"Shape of Scratch predicted probabilities: {scratch_predictions_proba.shape}")

    if np.any(np.isnan(scratch_predictions_proba)) or np.any(np.isinf(scratch_predictions_proba)):
        print("Error: NaN or Inf found in scratch model predictions. This indicates a numerical issue.")
    elif np.allclose(keras_predictions_proba, scratch_predictions_proba, atol=1e-5): 
        print("\nSUCCESS: Keras and From-Scratch model probability outputs are very close!")
    else:
        print("\nWARNING: Keras and From-Scratch model probability outputs differ significantly.")
        diff = np.abs(keras_predictions_proba - scratch_predictions_proba)
        print(f"Max absolute difference in probabilities: {np.max(diff):.6e}")
        print(f"Mean absolute difference in probabilities: {np.mean(diff):.6e}")
        for i_sample in range(min(3, keras_predictions_proba.shape[0])): 
            print(f"Sample {i_sample}:")
            print(f"  Keras:  {keras_predictions_proba[i_sample]}")
            print(f"  Scratch:{scratch_predictions_proba[i_sample]}")
            print(f"  Diff:   {keras_predictions_proba[i_sample] - scratch_predictions_proba[i_sample]}")

    keras_predicted_labels = np.argmax(keras_predictions_proba, axis=1)
    scratch_predicted_labels = np.argmax(scratch_predictions_proba, axis=1)

    keras_f1 = f1_score(test_labels_raw, keras_predicted_labels, average='macro', zero_division=0)
    scratch_f1 = f1_score(test_labels_raw, scratch_predicted_labels, average='macro', zero_division=0)

    print(f"\nMacro F1 Score (Keras): {keras_f1:.4f}")
    print(f"Macro F1 Score (From-Scratch): {scratch_f1:.4f}")

    if np.array_equal(keras_predicted_labels, scratch_predicted_labels):
        print("SUCCESS: Predicted labels from Keras and Scratch models are identical.")
    else:
        mismatches = np.sum(keras_predicted_labels != scratch_predicted_labels)
        total_samples = len(test_labels_raw)
        print(f"WARNING: Predicted labels differ for {mismatches}/{total_samples} samples ({mismatches/total_samples*100:.2f}% mismatch).")


def main():
    parser = argparse.ArgumentParser(description="End-to-end text classification pipeline.")
    parser.add_argument(
        '--train',
        action='store_true',
        help="Run the full training pipeline with hyperparameter sweep."
    )
    parser.add_argument(
        '--test-scratch',
        type=str,
        metavar='MODEL_NAME_OR_CONFIG',
        help="Compare Keras forward pass with from-scratch implementation. "
             "Provide the model configuration name (e.g., 'layers_1_units_64_dir_unidirectional') OR 'best_model'."
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=None, 
        help="Batch size for from-scratch model inference during --test-scratch. "
             "Processes all test data at once if not set. Also used for Keras model.predict batch_size."
    )
    parser.add_argument(
        '--demo-backward',
        action='store_true',
        help="Run a demonstration of the backward pass on a small batch when using --test-scratch. "
             "This will compute gradients but not update weights for the main comparison."
    )

    args = parser.parse_args()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists("plots"):
        os.makedirs("plots")

    if args.train:
        print("Starting training pipeline...")
        run_training_pipeline()
    elif args.test_scratch:
        run_test_scratch(args.test_scratch, args.inference_batch_size, args.demo_backward)
    else:
        print("No action specified. Use --train or --test-scratch MODEL_NAME_OR_CONFIG.")
        parser.print_help()

if __name__ == '__main__':
    main()