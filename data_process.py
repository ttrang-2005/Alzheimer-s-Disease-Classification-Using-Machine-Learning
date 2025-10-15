import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

# Load and preprocess data
df = pd.read_csv("alzheimers_disease_data.csv")
df = df.drop(columns=["PatientID", "DoctorInCharge"])
y = df["Diagnosis"]
X = df.drop(columns=["Diagnosis"])

# Balance classes using undersampling
df_majority = df[df["Diagnosis"] == y.value_counts().idxmax()]
df_minority = df[df["Diagnosis"] == y.value_counts().idxmin()]
df_majority_downsampled = resample(df_majority,
                                  replace=False,
                                  n_samples=len(df_minority),
                                  random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
X_bal = df_balanced.drop(columns=["Diagnosis"])
y_bal = df_balanced["Diagnosis"]

# Visualize class distribution
label_counts = y_bal.value_counts()
plt.figure(figsize=(6, 4))
plt.bar(label_counts.index, label_counts.values, color=['blue', 'red'])
plt.title('Phân bố lớp sau khi cân bằng')
plt.xlabel('Chẩn đoán ')
plt.ylabel('Số lượng mẫu')
plt.xticks(label_counts.index, ['Không Alzheimer', 'Alzheimer'])
plt.show()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

