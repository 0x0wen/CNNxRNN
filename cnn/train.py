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
EPOCHS_MAIN_MODEL = 15 
EPOCHS_EXPERIMENT = 5  
VAL_SPLIT_RATIO = 0.2 

PLOTS_DIR = "plots_cnn"
WEIGHTS_DIR = "weights_cnn"
MAIN_MODEL_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cnn_main_model.weights.h5")

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
    model_name_detail = (f"L{config['num_conv_layers']}_"
                         f"F{'x'.join(map(str,config['filters_list']))}_"
                         f"K{'x'.join([''.join(map(str,k)) for k in config['kernel_sizes_list']])}_"
                         f"P{config['pooling_type']}_G{config['use_global_pooling']}")
    full_model_name = f"{experiment_name_prefix}_{model_name_detail}"
    
    print(f"\n===== MENJALANKAN: {full_model_name} =====")
    
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
        epochs=config.get('epochs', EPOCHS_EXPERIMENT), 
        validation_data=(x_val_data, y_val_data),
        verbose=1 
    )
    
    training_time = time.time() - start_time
    print(f"Waktu pelatihan ({full_model_name}): {training_time:.2f} detik")

    plot_training_history(history, model_name_detail, experiment_name_prefix)
    f1, acc = evaluate_and_get_f1(model, x_test_data, y_test_data, full_model_name, experiment_name_prefix)
    
    return {"model_name": full_model_name, "f1_score": f1, "accuracy": acc, "training_time": training_time, "model_instance": model}

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_cifar10()
    
    results_summary = []

    print("\n--- Melatih Model CNN (Bobot akan disimpan) ---")
    main_model_config = {
        "num_conv_layers": 2,
        "filters_list": [32, 64],
        "kernel_sizes_list": [(3,3), (3,3)],
        "pooling_type": 'max',
        "use_global_pooling": False,
        "epochs": EPOCHS_MAIN_MODEL 
    }
    main_model_experiment_results = run_experiment(main_model_config, x_train, y_train, x_val, y_val, x_test, y_test, "MainModel")
    main_keras_model_instance = main_model_experiment_results["model_instance"]
    
    print(f"\nMenyimpan bobot untuk model: {main_model_experiment_results['model_name']}")
    main_keras_model_instance.save_weights(MAIN_MODEL_WEIGHTS_PATH)
    print(f"Bobot model CNN disimpan di: {MAIN_MODEL_WEIGHTS_PATH}")

    results_summary.append({k: v for k, v in main_model_experiment_results.items() if k != "model_instance"})

    run_additional_experiments = False 
    if run_additional_experiments:
        print("\n\n===== MEMULAI EKSPERIMEN=====")
        base_filters_per_layer = 32
        base_kernel_size = (3,3)
        
        num_layer_variations = [1, 3]
        for num_layers in num_layer_variations:
            config = {
                "num_conv_layers": num_layers,
                "filters_list": [base_filters_per_layer * (2**i) for i in range(num_layers)], 
                "kernel_sizes_list": [base_kernel_size] * num_layers,
                "pooling_type": 'max', "use_global_pooling": False, "epochs": EPOCHS_EXPERIMENT
            }
            exp_results = run_experiment(config, x_train, y_train, x_val, y_val, x_test, y_test, "Exp1_NumLayers")
            results_summary.append({k: v for k, v in exp_results.items() if k != "model_instance"})

        filters_variations = [[16, 32], [64, 128]]
        for filters in filters_variations:
            config = {
                "num_conv_layers": 2, "filters_list": filters,
                "kernel_sizes_list": [base_kernel_size] * 2,
                "pooling_type": 'max', "use_global_pooling": False, "epochs": EPOCHS_EXPERIMENT
            }
            exp_results = run_experiment(config, x_train, y_train, x_val, y_val, x_test, y_test, "Exp2_NumFilters")
            results_summary.append({k: v for k, v in exp_results.items() if k != "model_instance"})

    if results_summary:
        print("\n\n===== RINGKASAN HASIL EKSPERIMEN (Macro F1-Score & Accuracy) =====")
        print(f"{'Model Name':<80} {'F1-Score':<10} {'Accuracy':<10} {'Train Time (s)':<15}")
        print("-" * 115)
        for res in results_summary:
            print(f"{res['model_name']:<80} {res['f1_score']:.<5} {res['accuracy']:.<5} {res['training_time']:.<8}")
        print("-" * 115)

    print("\nPelatihan dan semua eksperimen (jika dijalankan) selesai.")
    print(f"Bobot model utama yang akan digunakan untuk implementasi 'from scratch' telah disimpan di: {MAIN_MODEL_WEIGHTS_PATH}")