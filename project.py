import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from joblib import parallel_backend
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from torch.amp import autocast, GradScaler

# Set better styling for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Color palette for consistent visualization
COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#d35400']

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"{title.center(60)}")
    print(f"{'=' * 60}")

def evaluate_model(y_true, y_pred, model_name, training_time, device="N/A"):
    """Evaluate model and return metrics in a structured format"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- {model_name} Performance Metrics ({device}) ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return acc, f1, training_time

def train_pytorch_model(X_train, y_train, X_test, y_test, device_name="cpu"):
    """Train PyTorch model on specified device"""
    device = torch.device(device_name)
    print(f"Training PyTorch model on: {device}")
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    # Move training data to device
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    class BinaryClassifier(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.model(x)

    model = BinaryClassifier(X_train.shape[1]).to(device)
    pos_weight = torch.tensor([sum(y_train==0)/sum(y_train==1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Initialize scaler for mixed precision training (only used for CUDA)
    scaler = GradScaler() if device_name == "cuda" else None

    train_losses, val_f1s = [], []
    best_f1 = 0
    patience_counter = 0
    max_patience = 5
    start_time = time.time()

    print("Training progress:")
    for epoch in range(25):
        model.train()
        total_loss = 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            
            # Use mixed precision for CUDA, standard precision for CPU
            if device_name == "cuda":
                with autocast(device_type='cuda'):  # Fixed: Specify device_type='cuda'
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_tensor)
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).cpu().numpy().astype(int)
            f1_val = f1_score(y_test, val_preds)
            val_f1s.append(f1_val)
            
            # Early stopping logic
            if f1_val > best_f1:
                best_f1 = f1_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience and epoch > 10:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print(f"Epoch {epoch+1:2d}/{25} - Loss: {avg_loss:.4f}, F1: {f1_val:.4f}" + 
              (", Best!" if f1_val == best_f1 else ""))

    training_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_test_tensor)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).cpu().numpy().astype(int)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'PyTorch Training Loss ({device_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_f1s)
    plt.title(f'PyTorch Validation F1 Score ({device_name})')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(f'pytorch_training_{device_name}.png')
    
    return val_preds, training_time

# Load dataset
print_section_header("DATA PREPARATION")
df = pd.read_csv('data.csv')
print("Dataset shape:", df.shape)
print("Missing values before handling:\n", df.isnull().sum())

# Fill missing values (median for numerical, mode for categorical)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
print("\nMissing values after handling:\n", df.isnull().sum())

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
print(f"\nCategorical columns encoded: {len(categorical_cols)}")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply SMOTE to balance dataset
X, y = SMOTE(random_state=42).fit_resample(X, y)
print(f"\nClass distribution after SMOTE:\n{pd.Series(y).value_counts()}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Feature dimensionality: {X_train.shape[1]}")

# Dictionary to store results for comparison
model_results = {}

# ==== Random Forest (Baseline) ====
print_section_header("RANDOM FOREST (BASELINE)")
start_rf_cpu = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
time_rf_cpu = time.time() - start_rf_cpu

acc_rf, f1_rf, _ = evaluate_model(y_test, y_pred_rf, "Random Forest", time_rf_cpu, "CPU")
model_results["Random Forest (CPU)"] = {"accuracy": acc_rf, "f1": f1_rf, "time": time_rf_cpu, "device": "CPU"}


# ==== Random Forest GPU ====
print_section_header("RANDOM FOREST GPU")

# Import only once before the timing starts
import cupy as cp
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF

# Move data to GPU before starting the timer
X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
X_test_gpu = cp.asarray(X_test, dtype=cp.float32)

# Start timing after data is already on GPU
start_rf_gpu = time.time()

# Initialize GPU random forest with optimized parameters
rf = cuRF(
    n_estimators=100,       # Match CPU's default 
    max_batch_size=4096,    # Process larger batches
    n_streams=2,            # Increase parallelism
    random_state=42,
    max_depth=16            # Limit tree depth for faster training
)

rf.fit(X_train_gpu, y_train_gpu)
y_pred_rf_gpu = rf.predict(X_test_gpu)

# Include prediction time but not data transfer time in benchmark
time_rf_gpu = time.time() - start_rf_gpu

# Convert predictions back to CPU after timing
y_pred_rf = y_pred_rf_gpu.get()

acc_rf, f1_rf, _ = evaluate_model(y_test, y_pred_rf, "Random Forest", time_rf_gpu, "GPU")
model_results["Random Forest (GPU)"] = {"accuracy": acc_rf, "f1": f1_rf, "time": time_rf_gpu, "device": "GPU"}


# Check if CUDA is available for GPU models
cuda_available = torch.cuda.is_available()
if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU detected: {gpu_name}")
else:
    gpu_name = "N/A"
    print("\nNo GPU detected, will run all models on CPU")

# ==== PyTorch on CPU ====
print_section_header("PYTORCH ON CPU")
y_pred_pytorch_cpu, time_pytorch_cpu = train_pytorch_model(X_train, y_train, X_test, y_test, "cpu")
acc_pytorch_cpu, f1_pytorch_cpu, _ = evaluate_model(y_test, y_pred_pytorch_cpu, "PyTorch", time_pytorch_cpu, "CPU")
model_results["PyTorch (CPU)"] = {"accuracy": acc_pytorch_cpu, "f1": f1_pytorch_cpu, "time": time_pytorch_cpu, "device": "CPU"}

# ==== PyTorch on GPU ====
if cuda_available:
    print_section_header("PYTORCH ON GPU")
    y_pred_pytorch_gpu, time_pytorch_gpu = train_pytorch_model(X_train, y_train, X_test, y_test, "cuda")
    acc_pytorch_gpu, f1_pytorch_gpu, _ = evaluate_model(y_test, y_pred_pytorch_gpu, "PyTorch", time_pytorch_gpu, "GPU")
    model_results["PyTorch (GPU)"] = {"accuracy": acc_pytorch_gpu, "f1": f1_pytorch_gpu, "time": time_pytorch_gpu, "device": "GPU"}

# ==== XGBoost on CPU ====
print_section_header("XGBOOST ON CPU")
start_xgb_cpu = time.time()
xgb_cpu = XGBClassifier(
    tree_method="hist", 
    device="cpu",
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    random_state=42
)
xgb_cpu.fit(X_train, y_train)
y_pred_xgb_cpu = xgb_cpu.predict(X_test)
time_xgb_cpu = time.time() - start_xgb_cpu

acc_xgb_cpu, f1_xgb_cpu, _ = evaluate_model(y_test, y_pred_xgb_cpu, "XGBoost", time_xgb_cpu, "CPU")
model_results["XGBoost (CPU)"] = {"accuracy": acc_xgb_cpu, "f1": f1_xgb_cpu, "time": time_xgb_cpu, "device": "CPU"}

# ==== XGBoost on GPU ====
if cuda_available:
    print_section_header("XGBOOST ON GPU")
    start_xgb_gpu = time.time()
    xgb_gpu = XGBClassifier(
        tree_method="gpu_hist", 
        device="cuda",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        random_state=42
    )
    xgb_gpu.fit(X_train, y_train)
    y_pred_xgb_gpu = xgb_gpu.predict(X_test)
    time_xgb_gpu = time.time() - start_xgb_gpu

    acc_xgb_gpu, f1_xgb_gpu, _ = evaluate_model(y_test, y_pred_xgb_gpu, "XGBoost", time_xgb_gpu, "GPU")
    model_results["XGBoost (GPU)"] = {"accuracy": acc_xgb_gpu, "f1": f1_xgb_gpu, "time": time_xgb_gpu, "device": "GPU"}

# ==== CatBoost on CPU ====
print_section_header("CATBOOST ON CPU")
start_cat_cpu = time.time()
cat_cpu = CatBoostClassifier(
    task_type="CPU",
    verbose=0,
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42
)
cat_cpu.fit(X_train, y_train)
y_pred_cat_cpu = cat_cpu.predict(X_test)
time_cat_cpu = time.time() - start_cat_cpu

acc_cat_cpu, f1_cat_cpu, _ = evaluate_model(y_test, y_pred_cat_cpu, "CatBoost", time_cat_cpu, "CPU")
model_results["CatBoost (CPU)"] = {"accuracy": acc_cat_cpu, "f1": f1_cat_cpu, "time": time_cat_cpu, "device": "CPU"}

# ==== CatBoost on GPU ====
if cuda_available:
    print_section_header("CATBOOST ON GPU")
    start_cat_gpu = time.time()
    cat_gpu = CatBoostClassifier(
        task_type="GPU",
        verbose=0,
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_seed=42
    )
    cat_gpu.fit(X_train, y_train)
    y_pred_cat_gpu = cat_gpu.predict(X_test)
    time_cat_gpu = time.time() - start_cat_gpu

    acc_cat_gpu, f1_cat_gpu, _ = evaluate_model(y_test, y_pred_cat_gpu, "CatBoost", time_cat_gpu, "GPU")
    model_results["CatBoost (GPU)"] = {"accuracy": acc_cat_gpu, "f1": f1_cat_gpu, "time": time_cat_gpu, "device": "GPU"}

# ==== Performance Summary ====
print_section_header("PERFORMANCE SUMMARY")

# Create DataFrame for comparison
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Device': [model_results[m]['device'] for m in model_results],
    'Accuracy': [model_results[m]['accuracy'] for m in model_results],
    'F1 Score': [model_results[m]['f1'] for m in model_results],
    'Time (s)': [model_results[m]['time'] for m in model_results]
})

# Sort by device then by time
comparison_df = comparison_df.sort_values(['Device', 'Time (s)'])
print(comparison_df.to_string(index=False))

# Calculate speedups for CPU vs GPU versions
if cuda_available:
    print("\nCPU vs GPU Speedup Analysis:")
    print("-" * 50)
    
    # Random Forest speedup
    rf_speedup = time_rf_cpu / time_rf_gpu
    print(f"Random Forest: CPU time: {time_rf_cpu:.2f}s, GPU time: {time_rf_gpu:.2f}s")
    print(f"          Speedup: {rf_speedup:.2f}x ({(rf_speedup-1)*100:.2f}% faster on GPU)")
    
    # PyTorch speedup
    pytorch_speedup = time_pytorch_cpu / time_pytorch_gpu
    print(f"PyTorch:  CPU time: {time_pytorch_cpu:.2f}s, GPU time: {time_pytorch_gpu:.2f}s")
    print(f"          Speedup: {pytorch_speedup:.2f}x ({(pytorch_speedup-1)*100:.2f}% faster on GPU)")
    
    # XGBoost speedup
    xgboost_speedup = time_xgb_cpu / time_xgb_gpu
    print(f"XGBoost:  CPU time: {time_xgb_cpu:.2f}s, GPU time: {time_xgb_gpu:.2f}s")
    print(f"          Speedup: {xgboost_speedup:.2f}x ({(xgboost_speedup-1)*100:.2f}% faster on GPU)")
    
    # CatBoost speedup
    catboost_speedup = time_cat_cpu / time_cat_gpu
    print(f"CatBoost: CPU time: {time_cat_cpu:.2f}s, GPU time: {time_cat_gpu:.2f}s")
    print(f"          Speedup: {catboost_speedup:.2f}x ({(catboost_speedup-1)*100:.2f}% faster on GPU)")

    # Overall speedup
    total_cpu_time = time_rf_cpu + time_pytorch_cpu + time_xgb_cpu + time_cat_cpu
    total_gpu_time = time_rf_gpu + time_pytorch_gpu + time_xgb_gpu + time_cat_gpu
    overall_speedup = total_cpu_time / total_gpu_time
    print(f"\nOverall:   CPU time: {total_cpu_time:.2f}s, GPU time: {total_gpu_time:.2f}s")
    print(f"          Speedup: {overall_speedup:.2f}x ({(overall_speedup-1)*100:.2f}% faster on GPU)")
    
# Create comparison plots
plt.figure(figsize=(15, 10))

# Training Time Comparison (by device)
plt.subplot(2, 2, 1)
sns.barplot(x='Model', y='Time (s)', hue='Device', data=comparison_df, palette={'CPU': '#3498db', 'GPU': '#e74c3c'})
plt.title('Training Time by Model and Device')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45, ha='right')
plt.yscale('log')  # Log scale for better visualization of time differences
plt.legend(title='Device')
# Accuracy Comparison
plt.subplot(2, 2, 2)
sns.barplot(x='Model', y='Accuracy', hue='Device', data=comparison_df, palette={'CPU': '#3498db', 'GPU': '#e74c3c'})
plt.title('Accuracy by Model and Device')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Device')
# F1 Score Comparison
plt.subplot(2, 2, 3)
sns.barplot(x='Model', y='F1 Score', hue='Device', data=comparison_df, palette={'CPU': '#3498db', 'GPU': '#e74c3c'})
plt.title('F1 Score by Model and Device')
plt.ylabel('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Device')
# Speedup Visualization (if GPU is available)
if cuda_available:
    plt.subplot(2, 2, 4)
    models = ['Random Forest', 'PyTorch', 'XGBoost', 'CatBoost', 'Overall']
    rf_speedup = model_results["Random Forest (CPU)"]["time"] / model_results["Random Forest (GPU)"]["time"]
    speedups = [rf_speedup, pytorch_speedup, xgboost_speedup, catboost_speedup, overall_speedup]
    colors = ['#d35400', '#1abc9c', '#f39c12', '#9b59b6', '#2ecc71']
    
    bars = plt.bar(models, speedups, color=colors)
    plt.title('GPU Speedup Relative to CPU (higher is better)')
    plt.ylabel('Speedup Factor (x)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add speedup values on top of bars
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.1, f'{v:.2f}x', ha='center')

plt.tight_layout()
plt.savefig('cpu_vs_gpu_comparison.png')
print("\nVisualization saved as 'cpu_vs_gpu_comparison.png'")

# ==== System Information ====
print_section_header("SYSTEM INFORMATION")
print(f"PyTorch Version: {torch.__version__}")
if cuda_available:
    print(f"CUDA Device: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA not available, all models ran on CPU")

print("\nAnalysis Complete!")