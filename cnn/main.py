import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

from cnn import CNNModel, Conv2D, ReLU, MaxPooling2D, AveragePooling2D, Flatten, Dense, SoftmaxActivation
from train import create_keras_cnn_model, load_and_preprocess_cifar10 

MAIN_MODEL_WEIGHTS_PATH = os.path.join("weights_cnn", "cnn_main_model.weights.h5")

MAIN_MODEL_ARCHITECTURE_CONFIG = {
    "num_conv_layers": 2,
    "filters_list": [32, 64],
    "kernel_sizes_list": [(3,3), (3,3)],
    "pooling_type": 'max', 
    "use_global_pooling": False, 
    "model_name_suffix": "MainModelForInference" 
}

INPUT_SHAPE_CHANNELS_LAST = (32, 32, 3)
INPUT_SHAPE_CHANNELS_FIRST = (3, 32, 32)
NUM_CLASSES = 10
CLASS_NAMES_CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DENSE_LAYER_UNITS = 128 

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()
    plt.close()

def get_keras_layer_output_shape(model, layer_name_to_find):
    found_layer_instance = None
    for layer_in_model in model.layers:
        if layer_in_model.name == layer_name_to_find:
            found_layer_instance = layer_in_model
            break
    
    if found_layer_instance is None:
        print(f"ERROR: Layer Keras dengan nama '{layer_name_to_find}' tidak ditemukan.")
        print("Nama layer yang tersedia di model Keras:")
        for l_obj in model.layers:
            print(f"  - {l_obj.name} (Type: {l_obj.__class__.__name__})")
        raise ValueError(f"Layer '{layer_name_to_find}' tidak ditemukan di model Keras.")

    if hasattr(found_layer_instance, 'output_shape') and found_layer_instance.output_shape is not None:
        return found_layer_instance.output_shape
    else: 
        if hasattr(found_layer_instance, 'output') and hasattr(found_layer_instance.output, 'shape'):
            shape_val = found_layer_instance.output.shape
            if hasattr(shape_val, 'as_list'): 
                return tuple(shape_val.as_list())
            elif isinstance(shape_val, tuple): 
                return shape_val
        raise AttributeError(f"Layer {found_layer_instance.name} tidak memiliki 'output_shape' atau 'output.shape' yang valid.")


if __name__ == "__main__":
    (_, _), (_, _), (x_test_keras, y_test_keras) = load_and_preprocess_cifar10()
    print(f"\nData uji dimuat: x_test_keras shape: {x_test_keras.shape}, y_test_keras shape: {y_test_keras.shape}")

    print("\n--- Membuat Ulang Model Keras, Build, dan Memuat Bobot ---")
    if not os.path.exists(MAIN_MODEL_WEIGHTS_PATH):
        print(f"ERROR: File bobot Keras tidak ditemukan di {MAIN_MODEL_WEIGHTS_PATH}")
        print("Pastikan Anda telah menjalankan train.py dan menyimpan bobot model utama.")
        exit()

    keras_model_reference = create_keras_cnn_model(**MAIN_MODEL_ARCHITECTURE_CONFIG)
    keras_model_reference.build(input_shape=(None, *INPUT_SHAPE_CHANNELS_LAST))
    print("Model Keras di-build secara eksplisit.")
    keras_model_reference.load_weights(MAIN_MODEL_WEIGHTS_PATH)
    print("Model Keras dan bobot berhasil dimuat sebagai referensi.")
    keras_model_reference.summary()
    
    print("\n--- Menginisialisasi Model CNN From Scratch dengan Input Shapes dari Keras ---")
    
    scratch_layers = []
    cfg = MAIN_MODEL_ARCHITECTURE_CONFIG 

    input_depth_conv_1 = INPUT_SHAPE_CHANNELS_LAST[-1] 
    print(f"Menentukan input_depth untuk scratch_conv_1: {input_depth_conv_1}")

    output_shape_keras_pool_1 = get_keras_layer_output_shape(keras_model_reference, "pool_1")
    input_depth_conv_2 = output_shape_keras_pool_1[-1] 
    print(f"Menentukan input_depth untuk scratch_conv_2 berdasarkan output Keras pool_1 ({output_shape_keras_pool_1}): {input_depth_conv_2}")

    if cfg["use_global_pooling"]:
        output_shape_keras_pooling_final = get_keras_layer_output_shape(keras_model_reference, "global_avg_pool_1")
    else:
        output_shape_keras_pooling_final = get_keras_layer_output_shape(keras_model_reference, "flatten_1")
    input_size_dense_1 = output_shape_keras_pooling_final[-1] 
    print(f"Menentukan input_size untuk scratch_dense_1 berdasarkan output Keras flatten_1 ({output_shape_keras_pooling_final}): {input_size_dense_1}")

    output_shape_keras_dense_1 = get_keras_layer_output_shape(keras_model_reference, "dense_1")
    input_size_output_dense = output_shape_keras_dense_1[-1] 
    print(f"Menentukan input_size untuk scratch_output_dense_1 berdasarkan output Keras dense_1 ({output_shape_keras_dense_1}): {input_size_output_dense}")

    scratch_layers.append(Conv2D(num_filters=cfg['filters_list'][0], filter_size=cfg['kernel_sizes_list'][0], 
                                 input_shape_depth=input_depth_conv_1, padding='same', name="scratch_conv_1"))
    scratch_layers.append(ReLU(name="scratch_relu_1"))
    if cfg['pooling_type'] == 'max':
        scratch_layers.append(MaxPooling2D(pool_size=(2,2), padding='valid', name="scratch_pool_1")) 
    elif cfg['pooling_type'] == 'average':
        scratch_layers.append(AveragePooling2D(pool_size=(2,2), padding='valid', name="scratch_pool_1"))

    if cfg['num_conv_layers'] >= 2:
        scratch_layers.append(Conv2D(num_filters=cfg['filters_list'][1], filter_size=cfg['kernel_sizes_list'][1],
                                     input_shape_depth=input_depth_conv_2, padding='same', name="scratch_conv_2"))
        scratch_layers.append(ReLU(name="scratch_relu_2"))
        if cfg['pooling_type'] == 'max':
            scratch_layers.append(MaxPooling2D(pool_size=(2,2), padding='valid', name="scratch_pool_2"))
        elif cfg['pooling_type'] == 'average':
            scratch_layers.append(AveragePooling2D(pool_size=(2,2), padding='valid', name="scratch_pool_2"))
    
    if cfg["use_global_pooling"]:
        raise NotImplementedError("GlobalAveragePooling2D from scratch belum diimplementasikan di cnn.py. Set use_global_pooling=False.")
    else:
        scratch_layers.append(Flatten(name="scratch_flatten_1"))

    scratch_layers.append(Dense(output_size=DENSE_LAYER_UNITS, input_size=input_size_dense_1, name="scratch_dense_1"))
    scratch_layers.append(ReLU(name="scratch_relu_3")) 

    scratch_layers.append(Dense(output_size=NUM_CLASSES, input_size=input_size_output_dense, name="scratch_output_dense_1"))
    scratch_layers.append(SoftmaxActivation(name="scratch_softmax_1"))

    cnn_model_scratch = CNNModel(layers=scratch_layers)
    
    print("\n--- Memuat Bobot Keras ke Model From Scratch ---")
    cnn_model_scratch.load_keras_weights(keras_model_reference) 
    print("Bobot Keras berhasil dimuat ke model from scratch.")
    
    print("\nMelakukan forward pass pada model scratch...")
    dummy_input_shape = (1, *INPUT_SHAPE_CHANNELS_FIRST) 
    dummy_input_data = np.random.rand(*dummy_input_shape).astype(np.float32)
    try:
        _ = cnn_model_scratch.forward(dummy_input_data) 
        print("Forward pass berhasil.")
    except Exception as e:
        print(f"ERROR saat forward pass: {e}")
        print("Summary mungkin tidak menampilkan output shapes dengan benar.")
    cnn_model_scratch.summary() 

    print("\n--- Melakukan Inferensi dan Perbandingan ---")
    
    num_test_samples = 1000 
    x_test_sample_keras = x_test_keras[:num_test_samples].astype(np.float32) 
    y_test_sample_keras = y_test_keras[:num_test_samples] 

    x_test_sample_scratch = np.transpose(x_test_sample_keras, (0, 3, 1, 2)).astype(np.float32) 
    print(f"Shape data uji untuk Keras: {x_test_sample_keras.shape}") 
    print(f"Shape data uji untuk Scratch: {x_test_sample_scratch.shape}") 

    print("\nMemprediksi dengan model Keras...") 
    start_time_keras = time.time() 
    y_pred_proba_keras = keras_model_reference.predict(x_test_sample_keras, verbose=0)
    end_time_keras = time.time() 
    y_pred_keras = np.argmax(y_pred_proba_keras, axis=1) 
    print(f"Waktu inferensi Keras (untuk {num_test_samples} sampel): {end_time_keras - start_time_keras:.4f} detik") 

    print("\nMemprediksi dengan model From Scratch...") 
    start_time_scratch = time.time() 
    
    y_pred_proba_scratch = cnn_model_scratch.forward(x_test_sample_scratch) 
    end_time_scratch = time.time() 
    y_pred_scratch = np.argmax(y_pred_proba_scratch, axis=1) 
    print(f"Waktu inferensi From Scratch (untuk {num_test_samples} sampel): {end_time_scratch - start_time_scratch:.4f} detik") 

    print("\n--- Perbandingan Hasil ---") 
    comparison_tolerance = 1e-4

    print("\nPerbandingan output probabilitas (maks 5 sampel pertama):") 
    for i in range(min(5, num_test_samples)): 
        print(f"Sampel {i+1}:") 
        print(f"  Keras    : {y_pred_proba_keras[i].round(decimals=4)}") 
        print(f"  Scratch  : {y_pred_proba_scratch[i].round(decimals=4)}") 
        print(f"  Label Asli: {y_test_sample_keras[i][0]}")
        if np.allclose(y_pred_proba_keras[i], y_pred_proba_scratch[i], atol=comparison_tolerance, rtol=comparison_tolerance): 
            print("  Output probabilitas numerik: Mirip (np.allclose TRUE)") 
        else:
            print("  Output probabilitas numerik: BERBEDA (np.allclose FALSE)") 
            print(f"    Perbedaan absolut maks: {np.max(np.abs(y_pred_proba_keras[i] - y_pred_proba_scratch[i])):.2e}") 


    f1_keras = f1_score(y_test_sample_keras.flatten(), y_pred_keras.flatten(), average="macro") 
    f1_scratch = f1_score(y_test_sample_keras.flatten(), y_pred_scratch.flatten(), average="macro") 

    print(f"\nMacro F1-Score (pada {num_test_samples} sampel uji):") 
    print(f"  Keras Model    : {f1_keras:.6f}") 
    print(f"  Scratch Model  : {f1_scratch:.6f}") 

    correct_predictions_match = np.sum(y_pred_keras == y_pred_scratch) 
    print(f"\nJumlah prediksi kelas yang sama persis antara Keras dan Scratch: " 
          f"{correct_predictions_match}/{num_test_samples} " 
          f"({(correct_predictions_match/num_test_samples)*100:.2f}%)") 

    print("\nLaporan Klasifikasi Model Keras:") 
    print(classification_report(y_test_sample_keras.flatten(), y_pred_keras.flatten(), target_names=CLASS_NAMES_CIFAR10, zero_division=0)) 

    print("\nLaporan Klasifikasi Model From Scratch:") 
    print(classification_report(y_test_sample_keras.flatten(), y_pred_scratch.flatten(), target_names=CLASS_NAMES_CIFAR10, zero_division=0)) 
    
    plot_confusion_matrix(y_test_sample_keras.flatten(), y_pred_keras.flatten(), CLASS_NAMES_CIFAR10, title="CM Keras Model")
    plot_confusion_matrix(y_test_sample_keras.flatten(), y_pred_scratch.flatten(), CLASS_NAMES_CIFAR10, title="CM Scratch Model")