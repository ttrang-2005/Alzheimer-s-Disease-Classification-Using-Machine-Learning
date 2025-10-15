import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import joblib
from data_process import X_scaled, y_bal, scaler  # Giả sử dữ liệu đã được xử lý trước
# Huấn luyện và đánh giá mô hình với k từ 2 đến 10
print("\n====== Mô hình: Random Forest ======")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Lưu trữ độ chính xác trung bình cho từng k
avg_accuracies = {}
last_fold_losses = None  # Lưu log-loss cho k=10 để vẽ biểu đồ
best_k = 2
best_avg_acc = 0
best_metrics = None  # Lưu trữ các chỉ số tại k tốt nhất
best_confusion_matrices = []  # Lưu trữ ma trận nhầm lẫn tại k tốt nhất
best_train_idx, best_test_idx = None, None  # Lưu chỉ số của fold tốt nhất

for k in range(2, 11):
    print(f"\n--- Số fold (k) = {k} ---")
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_losses = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_confusion_matrices = []
    fold = 1

    for train_idx, test_idx in kf.split(X_scaled, y_bal):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_bal.iloc[train_idx], y_bal.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        y_prob = model.predict_proba(X_test)
        loss = log_loss(y_test, y_prob)
        fold_losses.append(loss)

        # Tính các chỉ số Precision, Recall, F1-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)

        # Tính ma trận nhầm lẫn
        cm = confusion_matrix(y_test, y_pred)
        fold_confusion_matrices.append(cm)

        print(f"  Fold {fold} - Độ chính xác: {acc:.4f}")
        fold += 1

    avg_acc = np.mean(fold_accuracies)
    avg_accuracies[k] = avg_acc
    print(f">>> Độ chính xác trung bình (k={k}): {avg_acc:.4f}")

    # Lưu giá trị k tốt nhất và chỉ số của fold đầu tiên
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_k = k
        best_metrics = {
            "Precision": np.mean(fold_precisions),
            "Recall": np.mean(fold_recalls),
            "F1-score": np.mean(fold_f1s)
        }
        best_confusion_matrices = fold_confusion_matrices
        best_train_idx, best_test_idx = list(kf.split(X_scaled, y_bal))[0]  # Lưu chỉ số fold đầu tiên

    # Lưu log-loss cho k=10 để vẽ biểu đồ
    if k == 10:
        last_fold_losses = fold_losses

# Tính ma trận nhầm lẫn trung bình tại k tốt nhất
avg_confusion_matrix = np.mean(best_confusion_matrices, axis=0)

# In các chỉ số tại k tốt nhất
print(f"\n>>> Giá trị k tốt nhất: {best_k}")
print(f">>> Độ chính xác trung bình cao nhất: {best_avg_acc:.4f}")
print(f">>> Precision trung bình (k={best_k}): {best_metrics['Precision']:.4f}")
print(f">>> Recall trung bình (k={best_k}): {best_metrics['Recall']:.4f}")
print(f">>> F1-score trung bình (k={best_k}): {best_metrics['F1-score']:.4f}")
print(f"\n>>> Ma trận nhầm lẫn trung bình tại k={best_k}:")
print(avg_confusion_matrix)

# Huấn luyện lại mô hình với k tốt nhất và tập huấn luyện tốt nhất
print(f"\n>>> Huấn luyện lại mô hình với k={best_k} trên tập huấn luyện tốt nhất")
kf = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=42)
X_train, X_test = X_scaled[best_train_idx], X_scaled[best_test_idx]
y_train, y_test = y_bal.iloc[best_train_idx], y_bal.iloc[best_test_idx]
model.fit(X_train, y_train)

# Lưu mô hình tốt nhất dưới dạng .pkl
model_filename = f'best_random_forest_model_k{best_k}.pkl'
joblib.dump(model, model_filename)
print(f">>> Mô hình tốt nhất đã được lưu vào '{model_filename}'")

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(avg_confusion_matrix, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Không mắc bệnh', 'Mắc bệnh'],
            yticklabels=['Không mắc bệnh', 'Mắc bệnh'])
plt.title(f"Ma trận nhầm lẫn trung bình tại k={best_k} (Random Forest)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Vẽ biểu đồ các chỉ số tại k tốt nhất
plt.figure(figsize=(8, 5))
metrics_names = list(best_metrics.keys())
metrics_values = list(best_metrics.values())
plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange'])
plt.title(f"Precision, Recall, F1-score tại k={best_k} (Random Forest)")
plt.ylabel("Giá trị")
plt.ylim(0, 1)
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('metrics_plot.png')
plt.close()

# Vẽ biểu đồ log-loss cho k=10
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(last_fold_losses)+1), last_fold_losses, marker='o', label="Random Forest (k=10)")
plt.title("Log-loss trên từng Fold (Random Forest, k=10)")
plt.xlabel("Fold")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('log_loss_plot.png')
plt.close()

# Dự đoán dữ liệu kiểm tra
test_data = {
    'PatientID': 4758,
    'Age': 75,
    'Gender': 0,
    'Ethnicity': 0,
    'EducationLevel': 1,
    'BMI': 18.776009409162835,
    'Smoking': 0,
    'AlcoholConsumption': 13.723825705512622,
    'PhysicalActivity': 4.649450668217012,
    'DietQuality': 8.341903191502704,
    'SleepQuality': 4.213209925103094,
    'FamilyHistoryAlzheimers': 0,
    'CardiovascularDisease': 0,
    'Diabetes': 0,
    'Depression': 0,
    'HeadInjury': 0,
    'Hypertension': 0,
    'SystolicBP': 117,
    'DiastolicBP': 119,
    'CholesterolTotal': 151.38313679710524,
    'CholesterolLDL': 69.62351040559693,
    'CholesterolHDL': 77.34681647712739,
    'CholesterolTriglycerides': 210.57086609648726,
    'MMSE': 10.139568430460008,
    'FunctionalAssessment': 3.4013735067187523,
    'MemoryComplaints': 0,
    'BehavioralProblems': 0,
    'ADL': 4.517248273101627,
    'Confusion': 1,
    'Disorientation': 0,
    'PersonalityChanges': 0,
    'DifficultyCompletingTasks': 0,
    'Forgetfulness': 0,
    'Diagnosis': 1,
    'DoctorInCharge': 'XXXConfid'
}

test_df = pd.DataFrame([test_data])
test_df = test_df.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"])
test_scaled = scaler.transform(test_df)

# Dự đoán với mô hình tốt nhất
prediction = model.predict(test_scaled)[0]
print(f"\nMô hình Random Forest (k={best_k}):")
print(f"  Nhãn dự đoán: {prediction} (0: Không mắc bệnh, 1: Mắc bệnh Alzheimer)")