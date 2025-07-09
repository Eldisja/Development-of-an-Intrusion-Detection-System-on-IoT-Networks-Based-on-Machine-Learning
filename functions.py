import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

import warnings

# Untuk mengabaikan warning penggunaan cpu
def ignore_warnings():
    warnings.filterwarnings("ignore", message="Device used : .*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet.callbacks")

def encoding_rt_iot22_6classes(X_train, X_test, y_train, y_test):
    # Konversi Series
    y_train = pd.Series(y_train).squeeze()
    y_test = pd.Series(y_test).squeeze()

    # Gabungkan untuk encoding label target
    y_combined = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    label_encoder_y = LabelEncoder()
    label_encoder_y.fit(y_combined)

    y_train_encoded = label_encoder_y.transform(y_train)
    y_test_encoded = label_encoder_y.transform(y_test)

    # Proses encoding fitur kategorikal
    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_cols_X = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols_X:
        le = LabelEncoder()
        # Memastikan hanya label yang dikenal diterapkan
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Menghindari error dengan mapping manual dan fill -1 untuk unknown
        X_test[col] = X_test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Scaling numerik
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        np.array(X_train_scaled, dtype=np.float32),
        np.array(X_test_scaled, dtype=np.float32),
        np.array(y_train_encoded, dtype=np.int64),
        np.array(y_test_encoded, dtype=np.int64)
    )

def plot_curve_logscale(fold_curve, metric_name="Validation LogLoss"):
    plt.figure(figsize=(10, 6))
    for i, curve in enumerate(fold_curve):
        plt.plot(curve, label=f'Fold {i+1}', alpha=0.5)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name} (log scale)')
    plt.title(f'{metric_name} per Epoch per Fold (Log Y Scale)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_curve_deltas(fold_curve, metric_name="Validation LogLoss"):
    plt.figure(figsize=(10, 6))
    base = np.array(fold_curve[0])

    for i in range(1, len(fold_curve)):
        current = np.array(fold_curve[i])
        min_len = min(len(base), len(current))

        delta = current[:min_len] - base[:min_len]
        plt.plot(delta, label=f'Fold {i+1} - Fold 1')

    plt.xlabel('Epoch')
    plt.ylabel(f'Delta {metric_name} vs Fold 1')
    plt.title(f'Difference in {metric_name} per Epoch (Fold i - Fold 1)')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_curve_transparent(fold_curve, metric_name="Validation LogLoss"):
    plt.figure(figsize=(10, 6))

    for i, curve in enumerate(fold_curve):
        plt.plot(curve, label=f'Fold {i+1}', alpha=0.5)

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Epoch per Fold (with transparency)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_curve(curves, metric_name="Metric"):
    plt.figure(figsize=(10, 6))
    for i, curve in enumerate(curves):
        plt.plot(curve, label=f'Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Epoch per Fold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_label_encoding(y_label, y_encoded):

    label_mapping_df = pd.DataFrame({
        'Label_Original': y_label,
        'Label_Encoded': y_encoded
    })
    print("===== Mapping Label Encoding (Attack_type) =====")
    print(label_mapping_df.drop_duplicates()
                         .sort_values(by='Label_Encoded')
                         .reset_index(drop=True)
                         .to_string(index=False))

def encoding_cic_iot_diad_6classes(X_train, X_test):
    # Salin agar tidak mengubah data asli
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Drop kolom bermasalah
    drop_cols = ['Flow Bytes/s', 'Flow Packets/s']
    X_train.drop(columns=[col for col in drop_cols if col in X_train.columns], inplace=True, errors='ignore')
    X_test.drop(columns=[col for col in drop_cols if col in X_test.columns], inplace=True, errors='ignore')

    # Label encoding untuk fitur kategorikal
    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        mapping_dict = {val: idx for idx, val in enumerate(le.classes_)}
        X_test[col] = X_test[col].astype(str).map(mapping_dict).fillna(-1).astype(int)

    # Scaling numerik
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tangani NaN dan inf
    X_train_np = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test_np = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return X_train_np, X_test_np

def encoding_cic_iot_2023_6classes(X_train, X_test, y_train, y_test):
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # Cek nilai tak valid dan ganti
    for df in [X_train_copy, X_test_copy]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encoding kategorikal
    categorical_cols = X_train_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_copy[col] = le.fit_transform(X_train_copy[col].astype(str))
        X_test_copy[col] = le.transform(X_test_copy[col].astype(str))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_copy)
    X_test_scaled = scaler.transform(X_test_copy)

    label_encoder = LabelEncoder()
    y_train_np = label_encoder.fit_transform(y_train)
    y_test_np = label_encoder.transform(y_test)

    return np.array(X_train_scaled, dtype=np.float32), np.array(X_test_scaled, dtype=np.float32), y_train_np, y_test_np

def plot_best_confusion_matrix_fastai(best_y_true, best_preds, best_fold):
    # Pastikan input berdimensi 1
    best_y_true = np.ravel(best_y_true)
    best_preds = np.ravel(best_preds)

    # Hitung confusion matrix
    cm = confusion_matrix(best_y_true, best_preds)

    # Gunakan label numerik dari hasil encoding
    labels = np.unique(np.concatenate([best_y_true, best_preds]))

    # Tampilkan confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45, cmap='Blues')

    plt.title(f'Confusion Matrix Fold Terbaik (Fold {best_fold}) - Label Encoding')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_best_confusion_matrix(best_y_true, best_preds, best_fold):

    cm = confusion_matrix(best_y_true, best_preds)
    best_cm_labels = np.unique(np.concatenate([best_y_true, best_preds]))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_cm_labels)
    disp.plot(xticks_rotation=45, cmap='Blues')

    plt.title(f'Confusion Matrix Fold Terbaik (Fold {best_fold})')
    plt.grid(False)
    plt.tight_layout()
    plt.show()



def plot_class_distribution(y_label, y_encoded, title):
    # Mapping: Tampilkan hasil label encoding berdampingan di console
    label_mapping_df = pd.DataFrame({
        'Label_Original': y_label,
        'Label_Encoded': y_encoded
    })

    # Hitung jumlah sampel berdasarkan Label_Encoded
    df_encoded_counts = pd.DataFrame({'Label_Encoded': y_encoded})
    count_per_label = df_encoded_counts['Label_Encoded'].value_counts().sort_index()

    # Buat DataFrame untuk plot
    df_plot = pd.DataFrame({
        'Label_Encoded': count_per_label.index,
        'Count': count_per_label.values
    })

    # Plot heatmap berdasarkan angka label encoded
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_plot.set_index('Label_Encoded').T, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel('Label Encoded')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()