import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from data_process import X_scaled, y_bal

# Định nghĩa các mô hình
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Lưu trữ độ chính xác trung bình và log-loss cho từng mô hình
avg_accuracies_all = {name: [] for name in models.keys()}  # Độ chính xác trung bình qua các k
log_losses_all = {name: [] for name in models.keys()}  # Log-loss tại k=10

# Huấn luyện và thu thập chỉ số với k từ 2 đến 10
for k in range(2, 11):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for name, model in models.items():
        fold_accuracies = []
        fold_losses = []
        
        for train_idx, test_idx in kf.split(X_scaled, y_bal):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_bal.iloc[train_idx], y_bal.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(acc)
            
            if k == 10:  # Chỉ tính log-loss tại k=10
                y_prob = model.predict_proba(X_test)
                loss = log_loss(y_test, y_prob)
                fold_losses.append(loss)
        
        avg_acc = np.mean(fold_accuracies)
        avg_accuracies_all[name].append(avg_acc)
        
        if k == 10:
            log_losses_all[name] = fold_losses

# 1. Biểu đồ đánh giá chung
# Biểu đồ 1: Độ chính xác trung bình qua các k
plt.figure(figsize=(10, 6))
for name, accuracies in avg_accuracies_all.items():
    plt.plot(range(2, 11), accuracies, marker='o', label=name)
plt.title("Độ chính xác trung bình qua các giá trị k (2-10)")
plt.xlabel("Số fold (k)")
plt.ylabel("Độ chính xác trung bình")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Biểu đồ 2: Log-loss tại k=10
plt.figure(figsize=(10, 6))
for name, losses in log_losses_all.items():
    plt.plot(range(1, len(losses)+1), losses, marker='o', label=name)
plt.title("Log-loss trên từng Fold tại k=10")
plt.xlabel("Fold")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Biểu đồ đánh giá riêng
# Biểu đồ 3: Độ chính xác trung bình qua các k cho từng mô hình
for name, accuracies in avg_accuracies_all.items():
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), accuracies, marker='o', color='b')
    plt.title(f"Độ chính xác trung bình qua các giá trị k (2-10) - {name}")
    plt.xlabel("Số fold (k)")
    plt.ylabel("Độ chính xác trung bình")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Biểu đồ 4: Log-loss tại k=10 cho từng mô hình
for name, losses in log_losses_all.items():
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', color='r')
    plt.title(f"Log-loss trên từng Fold tại k=10 - {name}")
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()