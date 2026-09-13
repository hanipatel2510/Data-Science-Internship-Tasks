import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# Deep Learning Framework: Keras / TensorFlow (Python 3.13 Compatible)
import keras
from keras import layers, callbacks, regularizers

# ----------------------------------------------------
# 1. SETUP DIRECTORIES AUTOMATICALLY
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("WEEK 5: DEEP LEARNING APPLICATION IN DATA SCIENCE")
print("Folders 'graphs' and 'output' verified/created.")
print("="*60)

# ----------------------------------------------------
# 2. LOAD & PREPARE DATASET
# ----------------------------------------------------
csv_path = os.path.join(BASE_DIR, "cleaned_titanic.csv")
if not os.path.exists(csv_path):
    # Fallback to week4 or week1 path if not present in current folder
    alt_path = os.path.join(BASE_DIR, "..", "week4-Task", "cleaned_titanic.csv")
    if os.path.exists(alt_path):
        csv_path = alt_path
    else:
        raise FileNotFoundError("cleaned_titanic.csv file nahi mili! Use week5-Task folder me paste karein.")

df = pd.read_csv(csv_path)
print(f"Dataset loaded successfully with shape: {df.shape}")

# Target column
target_col = 'Survived'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Drop identifier columns if present
drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df.columns]
data = df.drop(columns=drop_cols)

# Separate features & labels
X = data.drop(columns=[target_col])
y = data[target_col].values

# Convert categorical columns to dummy variables (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Features Shape: {X_train_scaled.shape}")
print(f"Testing Features Shape:  {X_test_scaled.shape}")

# ----------------------------------------------------
# 3. BUILD DEEP LEARNING NEURAL NETWORK ARCHITECTURE
# ----------------------------------------------------
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    layers.Dense(16, activation='relu'),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Save architecture summary to output folder
with open(os.path.join(OUTPUT_DIR, "model_architecture.txt"), "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("\nNeural Network Architecture built successfully.")

# ----------------------------------------------------
# 4. MODEL TRAINING WITH EARLY STOPPING
# ----------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------------------------------
# 5. EVALUATION & METRICS CALCULATION
# ----------------------------------------------------
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print("\n" + "="*50)
print(f"Deep Learning Test Accuracy:  {test_acc:.4f}")
print(f"Precision Score:             {precision:.4f}")
print(f"Recall Score:                {recall:.4f}")
print(f"F1 Score:                    {f1:.4f}")
print(f"ROC-AUC Score:               {roc_auc:.4f}")
print("="*50)

# Save Classification Report
cls_report = classification_report(y_test, y_pred)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("TITANIC SURVIVAL DEEP LEARNING (ANN) REPORT\n")
    f.write("="*50 + "\n")
    f.write(cls_report)
    f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")

# Save Metrics CSV
metrics_df = pd.DataFrame([{
    "Model": "Deep Neural Network (MLP)",
    "Accuracy": round(test_acc, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1_Score": round(f1, 4),
    "ROC_AUC": round(roc_auc, 4)
}])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "deep_learning_metrics.csv"), index=False)

# ----------------------------------------------------
# 6. GENERATE & SAVE GRAPHS
# ----------------------------------------------------
# Graph 1: Training & Validation History (Loss and Accuracy)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax[0].plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
ax[0].plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
ax[0].set_title('Model Loss over Epochs', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Binary Cross-Entropy Loss')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Accuracy plot
ax[1].plot(history.history['accuracy'], label='Train Accuracy', color='#2ca02c', linewidth=2)
ax[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='#d62728', linewidth=2, linestyle='--')
ax[1].set_title('Model Accuracy over Epochs', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "01_training_history.png"), dpi=300)
plt.close()

# Graph 2: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Did Not Survive (0)', 'Survived (1)'],
            yticklabels=['Did Not Survive (0)', 'Survived (1)'])
plt.title('Deep Learning Model Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "02_confusion_matrix.png"), dpi=300)
plt.close()

# Graph 3: ROC Curve
plt.figure(figsize=(6.5, 5))
plt.plot(fpr, tpr, color='#1F4E79', lw=2.5, label=f'ANN ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "03_roc_curve.png"), dpi=300)
plt.close()

print("\n" + "="*60)
print("SUCCESS: All graphs & outputs have been saved in:")
print(f"Graphs: {GRAPHS_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("="*60)