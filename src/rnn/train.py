import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from tensorflow.keras.layers import TextVectorization, Embedding, SimpleRNN, GRU, LSTM, Dropout, Dense, Bidirectional # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict, Any, Optional
import shutil

MAX_FEATURES = 20000 
SEQUENCE_LENGTH = 120 
EMBEDDING_DIM = 256
NUM_CLASSES = 3
RNN_TYPE = "SimpleRNN"

HP_NUM_RNN_LAYERS = [1, 2, 3]
HP_RNN_UNITS = [32, 64, 128]
HP_RNN_DIRECTIONALITY = ["unidirectional", "bidirectional"]

MODEL_DIR = "models"
PLOT_DIR = "plots"
VECTORIZER_VOCAB_FILE = "vectorizer_vocab.json"

def load_data(file_path: str) -> Optional[Tuple[List[str], np.ndarray]]:
    try:
        df = pd.read_csv(file_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"Error: CSV file {file_path} must contain 'text' and 'label' columns.")
            return None
        
        texts = df['text'].astype(str).tolist()
        
        le = LabelEncoder()
        labels = le.fit_transform(df['label'])
        
        if len(le.classes_) != NUM_CLASSES:
            print(f"Warning: Number of unique labels found ({len(le.classes_)}) does not match NUM_CLASSES ({NUM_CLASSES}).")
            print(f"Detected classes: {le.classes_}. Please verify NUM_CLASSES.")
        
        print(f"Loaded {len(texts)} samples from {file_path}. Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        return texts, np.array(labels)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def create_text_vectorizer(train_texts: List[str], vocab_config_name: str) -> TextVectorization:
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=SEQUENCE_LENGTH,
        name="text_vectorizer"
    )
    vectorizer.adapt(train_texts)
    
    vocab = vectorizer.get_vocabulary()
    config = vectorizer.get_config()
    
    vocab_data = {
        'vocabulary': vocab,
        'config': {
            'max_tokens': config['max_tokens'],
            'output_sequence_length': config['output_sequence_length'],
            'standardize': config['standardize'],
            'split': config['split'],
            'ngrams': config['ngrams'],
            'output_mode': config['output_mode'],
        }
    }
    
    vocab_save_path = os.path.join(MODEL_DIR, f"vectorizer_vocab_{vocab_config_name}.json")
    with open(vocab_save_path, 'w') as f:
        json.dump(vocab_data, f)
    print(f"TextVectorization vocabulary and config saved to {vocab_save_path}")
    return vectorizer

def build_model(
    vectorizer: TextVectorization,
    num_rnn_layers: int,
    rnn_units: int,
    rnn_direction: str,
    dropout_rate: float = 0.3
) -> Sequential:
    model = Sequential(name=f"model_layers_{num_rnn_layers}_units_{rnn_units}_dir_{rnn_direction}")
    
    model.add(vectorizer)
    
    model.add(Embedding(
        input_dim=MAX_FEATURES,
        output_dim=EMBEDDING_DIM,
        input_length=SEQUENCE_LENGTH,
        name="embedding"
    ))

    for i in range(num_rnn_layers):
        return_sequences = (i < num_rnn_layers - 1)
        name_suffix = f"_{i+1}" if num_rnn_layers > 1 else ""
        
        rnn_core_layer = SimpleRNN(
            units=rnn_units,
            return_sequences=return_sequences,
            name=f"{RNN_TYPE.lower()}{name_suffix}"
        )
        
        if rnn_direction == "bidirectional":
            model.add(Bidirectional(rnn_core_layer, name=f"bidirectional_{rnn_core_layer.name}"))
        else:
            model.add(rnn_core_layer)

    model.add(Dropout(dropout_rate, name="dropout"))
    
    model.add(Dense(NUM_CLASSES, activation="softmax", name="output_dense"))
    
    return model

def train_and_evaluate_model(
    model: Sequential,
    train_data: tf.data.Dataset,
    valid_data: tf.data.Dataset,
    test_texts: List[str],
    test_labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    model_config_name: str = "default"
) -> Tuple[Dict[str, Any], float]:
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model_save_path = os.path.join(MODEL_DIR, f"{model.name}.keras") 
    
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    print(f"\n--- Training model: {model.name} ---")

    try:
        model.build(input_shape=(None,)) 
        print(f"Model {model.name} built with input shape: {model.input_shape}")
        model.summary()
    except Exception as e:
        print(f"Note: Model.build() or model.summary() encountered an issue (often ignorable if using tf.data.Dataset): {e}")


    history_object = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    plot_metrics_curves(history_object.history, model_config_name)
    
    loaded_model_for_this_config = model
    if os.path.exists(model_save_path):
        print(f"Loading best version of {model.name} from {model_save_path} for final evaluation.")
        loaded_model_for_this_config = tf.keras.models.load_model(model_save_path, compile=False)
        loaded_model_for_this_config.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    else:
        print(f"Warning: Best model file {model_save_path} not found. Using model state at end of training for evaluation of {model.name}.")

    if not isinstance(loaded_model_for_this_config.layers[0], tf.keras.layers.TextVectorization):
        print("ERROR: The first layer of the loaded_model_for_this_config is NOT TextVectorization!")
    else:
        print(f"First layer of loaded_model_for_this_config ({loaded_model_for_this_config.name}) is: {loaded_model_for_this_config.layers[0].name} (type: {type(loaded_model_for_this_config.layers[0])})")


    print(f"Evaluating model {loaded_model_for_this_config.name} on test data...")
    if not all(isinstance(text, str) for text in test_texts):
        print("ERROR: test_texts contains non-string elements! Attempting to convert all to string.")
        test_texts = [str(t) for t in test_texts]

    test_predictions_proba = loaded_model_for_this_config.predict(tf.constant(test_texts, dtype=tf.string), batch_size=batch_size)
    
    test_predictions_labels = np.argmax(test_predictions_proba, axis=1)
    
    macro_f1 = f1_score(test_labels, test_predictions_labels, average='macro', zero_division=0)
    print(f"Test Macro F1-Score for {model.name}: {macro_f1:.4f}")
    
    return history_object.history, macro_f1

def plot_metrics_curves(history_dict: Dict[str, Any], plot_config_name: str, display_duration: int = 0.1):
    loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])
    accuracy = history_dict.get('accuracy', [])
    val_accuracy = history_dict.get('val_accuracy', [])
    
    if loss and accuracy:
        print(f"Final Training Loss: {loss[-1]:.4f}")
        print(f"Final Training Accuracy: {accuracy[-1]:.4f}")
    if val_loss and val_accuracy:
        print(f"Final Validation Loss: {val_loss[-1]:.4f}")
        print(f"Final Validation Accuracy: {val_accuracy[-1]:.4f}")

    if not loss or not val_loss:
        print(f"Warning: Loss or val_loss not found in history for {plot_config_name}. Skipping loss plot.")
    if not accuracy or not val_accuracy:
        print(f"Warning: Accuracy or val_accuracy not found in history for {plot_config_name}. Skipping accuracy plot.")

    epochs_range_loss = range(len(loss)) if loss else []
    epochs_range_accuracy = range(len(accuracy)) if accuracy else []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if loss and val_loss:
        ax1.plot(epochs_range_loss, loss, color='#1f77b4', linewidth=2, label='Training Loss')
        ax1.plot(epochs_range_loss, val_loss, color='#ff7f0e', linewidth=2, label='Validation Loss')
        ax1.set_title('Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    else:
        ax1.text(0.5, 0.5, 'Loss data not available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax1.transAxes)

    if accuracy and val_accuracy:
        ax2.plot(epochs_range_accuracy, accuracy, color='#1f77b4', linewidth=2, label='Training Accuracy')
        ax2.plot(epochs_range_accuracy, val_accuracy, color='#ff7f0e', linewidth=2, label='Validation Accuracy')
        ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, 'Accuracy data not available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax2.transAxes)

    plt.tight_layout()

    plot_save_path = os.path.join(PLOT_DIR, f"metrics_curves_{plot_config_name}.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics curves (Loss and Accuracy) saved to {plot_save_path}")

    
    
    if display_duration > 0:
        plt.show(block=False)
        print(f"Displaying plot for {display_duration} seconds...")
        plt.pause(display_duration)
        plt.close(fig)
        print("Plot closed.")
    else:
        plt.close(fig)


def load_and_preprocess_data_for_testing(
    for_scratch_test: bool = False,
    batch_size: int = 32
) -> Tuple[Optional[List[str]], Optional[np.ndarray],
           Optional[List[str]], Optional[np.ndarray],
           Optional[List[str]], Optional[np.ndarray],
           Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
    train_data_loaded = load_data('data/train.csv')
    valid_data_loaded = load_data('data/valid.csv')
    test_data_loaded = load_data('data/test.csv')

    if not train_data_loaded or not valid_data_loaded or not test_data_loaded:
        print("Error: Could not load one or more data files in load_and_preprocess_data_for_testing.")
        return None, None, None, None, None, None, None, None, None

    train_texts, train_labels = train_data_loaded
    valid_texts, valid_labels = valid_data_loaded
    test_texts_raw, test_labels_raw = test_data_loaded

    if for_scratch_test:
        return (train_texts, train_labels,
                valid_texts, valid_labels,
                test_texts_raw, test_labels_raw,
                None, None, None)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_texts, valid_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_texts_raw, test_labels_raw)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return (train_texts, train_labels,
            valid_texts, valid_labels,
            test_texts_raw, test_labels_raw,
            train_dataset, valid_dataset, test_dataset)


def run_training_pipeline():
    print("Starting Keras training pipeline...")

    raw_train_data = load_data('data/train.csv')
    raw_valid_data = load_data('data/valid.csv')
    raw_test_data = load_data('data/test.csv')

    if not raw_train_data or not raw_valid_data or not raw_test_data:
        print("Failed to load data. Aborting training.")
        return

    train_texts, train_labels = raw_train_data
    valid_texts, valid_labels = raw_valid_data
    test_texts_raw, test_labels_raw = raw_test_data
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))\
        .shuffle(buffer_size=len(train_texts))\
        .batch(32)\
        .prefetch(tf.data.AUTOTUNE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_texts, valid_labels))\
        .batch(32)\
        .prefetch(tf.data.AUTOTUNE)

    all_results: List[Dict[str, Any]] = []
    best_f1_overall = -1.0

    best_overall_model_path = os.path.join(MODEL_DIR, "best_model.keras") 
    best_overall_config_name = ""
    best_overall_config_dict: Dict[str, Any] = {}

    best_by_layers: Dict[int, Tuple[float, str, Dict[str, Any]]] = {n: (-1.0, "", {}) for n in HP_NUM_RNN_LAYERS}
    best_by_units: Dict[int, Tuple[float, str, Dict[str, Any]]] = {u: (-1.0, "", {}) for u in HP_RNN_UNITS}
    best_by_direction: Dict[str, Tuple[float, str, Dict[str, Any]]] = {d: (-1.0, "", {}) for d in HP_RNN_DIRECTIONALITY}

    print("Adapting shared TextVectorization layer on training data...")
    global_vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=SEQUENCE_LENGTH,
        name="shared_text_vectorizer"
    )
    global_vectorizer.adapt(train_texts)

    shared_vocab_data = {
        'vocabulary': global_vectorizer.get_vocabulary(),
        'config': global_vectorizer.get_config()
    }
    shared_vocab_path = os.path.join(MODEL_DIR, "shared_vectorizer_vocab.json")
    with open(shared_vocab_path, 'w') as f:
        json.dump(shared_vocab_data, f, indent=4)
    print(f"Shared TextVectorization vocabulary and config saved to {shared_vocab_path}")


    print("\n--- Starting Hyperparameter Sweep ---")
    for num_layers_hp in HP_NUM_RNN_LAYERS:
        for units_hp in HP_RNN_UNITS:
            for direction_hp in HP_RNN_DIRECTIONALITY:
                current_config_name = f"layers_{num_layers_hp}_units_{units_hp}_dir_{direction_hp}"
                print(f"\n>>>> CONFIGURATION: {current_config_name} (RNN Type: {RNN_TYPE}) <<<<")

                model = build_model(
                    vectorizer=global_vectorizer,
                    num_rnn_layers=num_layers_hp,
                    rnn_units=units_hp,
                    rnn_direction=direction_hp
                )
                
                history_dict, macro_f1_score = train_and_evaluate_model(
                    model=model,
                    train_data=train_dataset,
                    valid_data=valid_dataset,
                    test_texts=test_texts_raw,
                    test_labels=test_labels_raw,
                    epochs=20,
                    batch_size=32,
                    model_config_name=current_config_name 
                )
                
                result_entry = {
                    "config_name": current_config_name,
                    "num_rnn_layers": num_layers_hp,
                    "rnn_units": units_hp,
                    "rnn_direction": direction_hp,
                    "rnn_type": RNN_TYPE,
                    "macro_f1_score": macro_f1_score,
                    "training_loss": history_dict.get('loss', []),
                    "validation_loss": history_dict.get('val_loss', []),
                    "training_accuracy": history_dict.get('accuracy', []),
                    "validation_accuracy": history_dict.get('val_accuracy', [])
                }
                all_results.append(result_entry)

                if macro_f1_score > best_f1_overall:
                    best_f1_overall = macro_f1_score
                    best_overall_config_name = current_config_name
                    best_overall_config_dict = {
                        "num_rnn_layers": num_layers_hp,
                        "rnn_units": units_hp,
                        "rnn_direction": direction_hp,
                        "rnn_type": RNN_TYPE
                    }
                    
                    individual_model_path = os.path.join(MODEL_DIR, f"{model.name}.keras")
                    if os.path.exists(individual_model_path):
                        shutil.copyfile(individual_model_path, best_overall_model_path)
                        print(f"New best overall model '{individual_model_path}' copied to '{best_overall_model_path}' (Macro F1: {macro_f1_score:.4f})")
                        
                        best_model_config_json_path = os.path.join(MODEL_DIR, "best_model_config.json")
                        with open(best_model_config_json_path, 'w') as f:
                            json.dump(best_overall_config_dict, f, indent=4)
                        print(f"Best overall model configuration saved to {best_model_config_json_path}")
                    else:
                        print(f"Warning: Model file {individual_model_path} for new best config was not found. Cannot copy to {best_overall_model_path}.")

                if macro_f1_score > best_by_layers[num_layers_hp][0]:
                    best_by_layers[num_layers_hp] = (macro_f1_score, current_config_name, history_dict)
                if macro_f1_score > best_by_units[units_hp][0]:
                    best_by_units[units_hp] = (macro_f1_score, current_config_name, history_dict)
                if macro_f1_score > best_by_direction[direction_hp][0]:
                    best_by_direction[direction_hp] = (macro_f1_score, current_config_name, history_dict)

                tf.keras.backend.clear_session()

    print("\n--- Hyperparameter Sweep Results ---")
    all_results_sorted = sorted(all_results, key=lambda x: x['macro_f1_score'], reverse=True)
    
    for res in all_results_sorted:
        print(f"Config: {res['config_name']:<45} | RNN Type: {res['rnn_type']:<10} | Macro F1: {res['macro_f1_score']:.4f}")

    print(f"\nBest overall model config name: '{best_overall_config_name}' with Macro F1: {best_f1_overall:.4f}")
    print(f"Best overall model saved to: {best_overall_model_path}")
    print(f"Best overall model config saved to: {os.path.join(MODEL_DIR, 'best_model_config.json')}")


    print("\n--- Saving Best Hyperparameter Plots (Metrics Curves) ---")
    for n_layers, (f1, conf_name, hist_dict) in best_by_layers.items():
        if f1 > -1.0:
            plot_name = f"Best_{n_layers}_layer_f1"
            plot_metrics_curves(hist_dict, plot_name)
            print(f"Saved metrics plot for best {n_layers} layer(s) (config: {conf_name}, F1: {f1:.4f}) as metrics_curves_{plot_name}.png")

    for units_val, (f1, conf_name, hist_dict) in best_by_units.items():
        if f1 > -1.0:
            plot_name = f"Best_{units_val}_units_f1"
            plot_metrics_curves(hist_dict, plot_name)
            print(f"Saved metrics plot for best {units_val} units (config: {conf_name}, F1: {f1:.4f}) as metrics_curves_{plot_name}.png")

    for direction_val, (f1, conf_name, hist_dict) in best_by_direction.items():
        if f1 > -1.0:
            plot_name = f"Best_{direction_val}_f1"
            plot_metrics_curves(hist_dict, plot_name)
            print(f"Saved metrics plot for best {direction_val} RNN (config: {conf_name}, F1: {f1:.4f}) as metrics_curves_{plot_name}.png")

    results_summary_save_path = os.path.join(PLOT_DIR, "hyperparameter_sweep_summary.json")
    with open(results_summary_save_path, 'w') as f:
        json.dump(all_results_sorted, f, indent=4)
    print(f"Hyperparameter sweep summary saved to {results_summary_save_path}")

if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please place train.csv, valid.csv, and test.csv there.")

    required_files = ['data/train.csv', 'data/valid.csv', 'data/test.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing data file(s): {', '.join(missing_files)}")
        print("Please ensure train.csv, valid.csv, and test.csv are in the 'data' directory.")
    else:
        print("Running train.py directly for training pipeline...")
        run_training_pipeline()