import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 64
EPOCHS = 10
VAL_SPLIT_RATIO = 0.2 

PLOTS_DIR = "plots_cnn"
WEIGHTS_DIR = "weights_cnn"
BEST_MODEL_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cnn_best_model.weights.h5")

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

def load_and_preprocess_cifar10():
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=VAL_SPLIT_RATIO, random_state=42, stratify=y_train_full
    )
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_keras_cnn_model(
    num_conv_layers=2,
    filters_list=[32, 64], 
    kernel_sizes_list=[(3,3), (3,3)], 
    pooling_type='max',
    use_global_pooling=False, 
    model_name_suffix=""
    ):
    if len(filters_list) != num_conv_layers or len(kernel_sizes_list) != num_conv_layers:
        raise ValueError("Panjang filters_list dan kernel_sizes_list harus sama dengan num_conv_layers.")

    model = keras.Sequential(name=f"KerasCNN_{model_name_suffix}")
    model.add(layers.InputLayer(shape=INPUT_SHAPE))

    for i in range(num_conv_layers):
        model.add(layers.Conv2D(filters_list[i], kernel_sizes_list[i], activation="relu", padding="same", name=f"conv_{i+1}"))
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{i+1}"))
        elif pooling_type == 'average':
            model.add(layers.AveragePooling2D(pool_size=(2, 2), name=f"pool_{i+1}"))
        else:
            raise ValueError("pooling_type tidak valid.")

    if use_global_pooling:
        model.add(layers.GlobalAveragePooling2D(name="global_avg_pool_1"))
    else:
        model.add(layers.Flatten(name="flatten_1"))
    
    model.add(layers.Dense(128, activation="relu", name="dense_1")) 
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output_dense_1"))

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def plot_training_history(history, model_name, experiment_name_prefix):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - {model_name}\n({experiment_name_prefix})')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy - {model_name}\n({experiment_name_prefix})')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.tight_layout()
    plot_filename = os.path.join(PLOTS_DIR, f"{experiment_name_prefix}_{model_name}_history.png")
    plt.savefig(plot_filename)
    print(f"Plot pelatihan disimpan di {plot_filename}")
    plt.close()

def evaluate_and_get_f1(model, x_test_data, y_test_data, model_name, experiment_name_prefix):
    loss, accuracy = model.evaluate(x_test_data, y_test_data, verbose=0)
    y_pred_proba = model.predict(x_test_data, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1 = f1_score(y_test_data.flatten(), y_pred.flatten(), average="macro") 
    
    print(f"\n--- Evaluasi Model: {model_name} ({experiment_name_prefix}) ---")
    print(f"  Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    print(f"  Test Macro F1-Score: {f1:.4f}")
    return f1, accuracy

def run_experiment(config, x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data, experiment_name_prefix):
    if experiment_name_prefix == "Exp1_NumLayers":
        model_name_detail = f"Experiment1_NumberLayer_{config['num_conv_layers']}"
    elif experiment_name_prefix == "Exp2_NumFilters":
        model_name_detail = f"Experiment2_Filters_{'x'.join(map(str,config['filters_list']))}"
    elif experiment_name_prefix == "Exp3_KernelSize":
        kernel_size = config['kernel_sizes_list'][0][0]  
        model_name_detail = f"Experiment3_KernelSize_{kernel_size}x{kernel_size}"
    elif experiment_name_prefix == "Exp4_PoolingType":
        model_name_detail = f"Experiment4_PoolingType_{config['pooling_type']}"
    
    print(f"\n===== MENJALANKAN: {model_name_detail} =====")
    
    start_time = time.time()
    model = create_keras_cnn_model(
        num_conv_layers=config['num_conv_layers'],
        filters_list=config['filters_list'],
        kernel_sizes_list=config['kernel_sizes_list'],
        pooling_type=config['pooling_type'],
        use_global_pooling=config['use_global_pooling'],
        model_name_suffix=model_name_detail
    )
    
    history = model.fit(
        x_train_data, y_train_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val_data, y_val_data),
        verbose=1 
    )
    
    training_time = time.time() - start_time
    print(f"Waktu pelatihan ({model_name_detail}): {training_time:.2f} detik")

    plot_training_history(history, model_name_detail, experiment_name_prefix)
    f1, acc = evaluate_and_get_f1(model, x_test_data, y_test_data, model_name_detail, experiment_name_prefix)
    
    return {"model_name": model_name_detail, "f1_score": f1, "accuracy": acc, "training_time": training_time, "model_instance": model}

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_cifar10()
    
    results_summary = []
    best_f1_score = 0
    best_model = None

    print("\n===== Eksperimen 1: Analisis Pengaruh Jumlah Layer Konvolusi =====")
    print("Variasi: 1 layer, 2 layer, dan 3 layer konvolusi")
    for num_layers in [1, 2, 3]:
        config = {
            "num_conv_layers": num_layers,
            "filters_list": [32 * (2**i) for i in range(num_layers)], 
            "kernel_sizes_list": [(3,3)] * num_layers,
            "pooling_type": 'max',
            "use_global_pooling": False
        }
        exp_results = run_experiment(config, x_train, y_train, x_val, y_val, x_test, y_test, "Exp1_NumLayers")
        results_summary.append({k: v for k, v in exp_results.items() if k != "model_instance"})
        
        if exp_results["f1_score"] > best_f1_score:
            best_f1_score = exp_results["f1_score"]
            best_model = exp_results["model_instance"]

    print("\n===== Eksperimen 2: Analisis Pengaruh Jumlah Filter per Layer =====")
    print("Variasi: [16,32], [32,64], dan [64,128] filter")
    filter_variations = [
        [16, 32], 
        [32, 64],   
        [64, 128]   
    ]
    
    for filters in filter_variations:
        config = {
            "num_conv_layers": 2,
            "filters_list": filters,
            "kernel_sizes_list": [(3,3), (3,3)],
            "pooling_type": 'max',
            "use_global_pooling": False
        }
        exp_results = run_experiment(config, x_train, y_train, x_val, y_val, x_test, y_test, "Exp2_NumFilters")
        results_summary.append({k: v for k, v in exp_results.items() if k != "model_instance"})
        
        if exp_results["f1_score"] > best_f1_score:
            best_f1_score = exp_results["f1_score"]
            best_model = exp_results["model_instance"]

    print("\n===== Eksperimen 3: Analisis Pengaruh Ukuran Filter =====")
    print("Variasi: 3x3, 5x5, dan 7x7 kernel")
    kernel_variations = [
        [(3,3), (3,3)],
        [(5,5), (5,5)],
        [(7,7), (7,7)]
    ]
    
    for kernels in kernel_variations:
        config = {
            "num_conv_layers": 2,
            "filters_list": [32, 64],
            "kernel_sizes_list": kernels,
            "pooling_type": 'max',
            "use_global_pooling": False
        }
        exp_results = run_experiment(config, x_train, y_train, x_val, y_val, x_test, y_test, "Exp3_KernelSize")
        results_summary.append({k: v for k, v in exp_results.items() if k != "model_instance"})
        
        if exp_results["f1_score"] > best_f1_score:
            best_f1_score = exp_results["f1_score"]
            best_model = exp_results["model_instance"]

    print("\n===== Eksperimen 4: Analisis Pengaruh Jenis Pooling =====")
    print("Variasi: Max Pooling vs Average Pooling")
    for pooling_type in ['max', 'average']:
        config = {
            "num_conv_layers": 2,
            "filters_list": [32, 64],
            "kernel_sizes_list": [(3,3), (3,3)],
            "pooling_type": pooling_type,
            "use_global_pooling": False
        }
        exp_results = run_experiment(config, x_train, y_train, x_val, y_val, x_test, y_test, "Exp4_PoolingType")
        results_summary.append({k: v for k, v in exp_results.items() if k != "model_instance"})
        
        if exp_results["f1_score"] > best_f1_score:
            best_f1_score = exp_results["f1_score"]
            best_model = exp_results["model_instance"]

    if results_summary:
        print("\n\n===== RINGKASAN HASIL EKSPERIMEN =====")
        print("\nEksperimen 1: Pengaruh Jumlah Layer Konvolusi")
        print("-" * 115)
        for res in [r for r in results_summary if "NumberLayer" in r["model_name"]]:
            print(f"{res['model_name']:<80} F1-Score: {res['f1_score']:.4f} Accuracy: {res['accuracy']:.4f} Time: {res['training_time']:.2f}s")
        
        print("\nEksperimen 2: Pengaruh Jumlah Filter")
        print("-" * 115)
        for res in [r for r in results_summary if "Filters" in r["model_name"]]:
            print(f"{res['model_name']:<80} F1-Score: {res['f1_score']:.4f} Accuracy: {res['accuracy']:.4f} Time: {res['training_time']:.2f}s")
        
        print("\nEksperimen 3: Pengaruh Ukuran Filter")
        print("-" * 115)
        for res in [r for r in results_summary if "KernelSize" in r["model_name"]]:
            print(f"{res['model_name']:<80} F1-Score: {res['f1_score']:.4f} Accuracy: {res['accuracy']:.4f} Time: {res['training_time']:.2f}s")
        
        print("\nEksperimen 4: Pengaruh Jenis Pooling")
        print("-" * 115)
        for res in [r for r in results_summary if "PoolingType" in r["model_name"]]:
            print(f"{res['model_name']:<80} F1-Score: {res['f1_score']:.4f} Accuracy: {res['accuracy']:.4f} Time: {res['training_time']:.2f}s")
        print("-" * 115)

    if best_model is not None:
        print(f"\nMenyimpan model terbaik dengan F1-Score: {best_f1_score:.4f}")
        best_model.save_weights(BEST_MODEL_WEIGHTS_PATH)
        print(f"Model terbaik disimpan di: {BEST_MODEL_WEIGHTS_PATH}")