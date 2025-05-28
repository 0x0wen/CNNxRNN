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

from cnn import CNNModel, Conv2D, ReLU, MaxPooling2D, AveragePooling2D, Flatten, Dense, SoftmaxActivation, create_scratch_cnn_model
from train import create_keras_cnn_model, load_and_preprocess_cifar10

BEST_MODEL_WEIGHTS_PATH = os.path.join("weights_cnn", "cnn_best_model.weights.h5")

MAIN_MODEL_ARCHITECTURE_CONFIG = {
    "num_conv_layers": 3,
    "filters_list": [32, 64, 128],
    "kernel_sizes_list": [(3,3), (3,3), (3,3)],
    "pooling_type": 'max', 
    "use_global_pooling": False, 
    "model_name_suffix": "BestModelForInference" 
}

INPUT_SHAPE_CHANNELS_LAST = (32, 32, 3)
INPUT_SHAPE_CHANNELS_FIRST = (3, 32, 32)
NUM_CLASSES = 10
CLASS_NAMES_CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DENSE_LAYER_UNITS = 128 

if __name__ == "__main__":
    (_, _), (_, _), (x_test_keras, y_test_keras) = load_and_preprocess_cifar10()
    print(f"\nData uji diload: x_test_keras shape: {x_test_keras.shape}, y_test_keras shape: {y_test_keras.shape}")

    print("\n--- Build Ulang Model Keras dan Load Bobot Model Terbaik ---")
    if not os.path.exists(BEST_MODEL_WEIGHTS_PATH):
        print(f"ERROR: File bobot model terbaik tidak ditemukan di {BEST_MODEL_WEIGHTS_PATH}")
        print("Pastikan Anda telah menjalankan train.py untuk melakukan eksperimen dan menyimpan bobot model terbaik.")
        exit()

    keras_model_reference = create_keras_cnn_model(**MAIN_MODEL_ARCHITECTURE_CONFIG)
    keras_model_reference.build(input_shape=(None, *INPUT_SHAPE_CHANNELS_LAST))
    print("Model Keras di-build secara eksplisit.")
    keras_model_reference.load_weights(BEST_MODEL_WEIGHTS_PATH)
    print("Model Keras dan bobot terbaik berhasil diload sebagai referensi.")
    keras_model_reference.summary()
    
    print("\n--- Menginisialisasi Model CNN From Scratch ---")
    cnn_model_scratch = create_scratch_cnn_model(
        config=MAIN_MODEL_ARCHITECTURE_CONFIG,
        input_shape_channels_first=INPUT_SHAPE_CHANNELS_FIRST,
        keras_model_reference=keras_model_reference
    )
    
    print("\n--- Load Bobot Keras ke Model From Scratch ---")
    cnn_model_scratch.load_keras_weights(keras_model_reference) 
    print("Bobot Keras berhasil diload ke model from scratch.")
    
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
    
    num_test_samples = 10000 
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
    
    batch_size = 5000
    num_batches = (num_test_samples + batch_size - 1) // batch_size
    y_pred_proba_scratch = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_samples)
        batch_output = cnn_model_scratch.forward(x_test_sample_scratch[start_idx:end_idx])
        y_pred_proba_scratch.append(batch_output)
    
    y_pred_proba_scratch = np.vstack(y_pred_proba_scratch)
    end_time_scratch = time.time() 
    y_pred_scratch = np.argmax(y_pred_proba_scratch, axis=1) 
    print(f"Waktu inferensi From Scratch (untuk {num_test_samples} sampel): {end_time_scratch - start_time_scratch:.4f} detik")

    print("\n--- Perbandingan Hasil ---") 

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