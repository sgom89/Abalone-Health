from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'Abalone Data Set [2874].xlsx'
IMAGES_DIR = BASE_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()


# 1. Load Data
df = pd.read_excel(DATA_PATH)
print("--- Initial Data Info ---")
df.info()

# 2. Data Cleaning & Preprocessing 
# Handle missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Fill missing numerical values with median (robust to outliers)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle invalid categorical values
df = df[df['Sex'].isin(['M', 'F', 'I'])]
df['Sex'] = df['Sex'].astype(str)

# Outlier Detection  - Using IQR method
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
# Visualize outliers and cap them to avoid losing data
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
for col in num_cols:
    df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])
    df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])

# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Sex')
plt.title('Distribution of Abalone Sex')
save_plot('eda_sex_dist.png')

plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Sex', y='Spots')
plt.title('Spots Distribution by Sex')
save_plot('eda_spots_by_sex.png')

plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
save_plot('eda_corr.png')

# 4. Regression Analysis  - Predicting 'Spots' (Health Proxy)
# A LASSO model was built for each Sex to compare feature importance
print("\n--- Regression Analysis (LASSO) ---")
scaler = StandardScaler()

for sex in ['M', 'F', 'I']:
    subset = df[df['Sex'] == sex]
    X = subset.drop(['Sex', 'Spots'], axis=1)
    y = subset['Spots']
    
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    
    print(f"\nLASSO Regression for {sex}:")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    
    importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso.coef_})
    importance = importance.sort_values(by='Coefficient', ascending=False)
    print("Feature Coefficients:")
    print(importance)

# 5. Classification Analysis - Predicting 'Sex'
print("\n--- Classification Analysis ---")
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex']) # F:0, I:1, M:2

X_class = df.drop(['Sex', 'Sex_encoded'], axis=1)
y_class = df['Sex_encoded']

X_class_scaled = scaler.fit_transform(X_class)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_c, y_train_c)
y_pred_dt = dt.predict(X_test_c)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test_c, y_pred_dt, target_names=le.classes_))

# Plot Decision Tree
plt.figure(figsize=(24, 12))
plot_tree(dt, feature_names=X_class.columns, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree for Sex Classification')
save_plot('decision_tree.png')

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_c, y_train_c)
y_pred_lr = lr.predict(X_test_c)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_c, y_pred_lr, target_names=le.classes_))

# 6. Clustering Analysis  - K-Means
print("\n--- Clustering Analysis ---")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_class_scaled)

# Check how clusters align with Sex
cluster_sex_crosstab = pd.crosstab(df['Cluster'], df['Sex'])
print("\nK-Means Clusters vs Sex:")
print(cluster_sex_crosstab)

print(f"\nAnalysis Complete. Plots saved to: {IMAGES_DIR}")
