import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

sns.set(style="whitegrid")

df = pd.read_csv("data.csv", sep=';')
df.head()

print(df.shape)
df.info()
df.describe().T

print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nJumlah duplikat:", df.duplicated().sum())
print("\nDistribusi target:\n", df['Status'].value_counts())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

df['Status'].value_counts().plot(kind='bar', ax=axes[0,0], title='Distribusi Status')

sns.barplot(x='Debtor', y=(df['Status']=='Dropout').astype(int), data=df, ax=axes[0,1])
axes[0,1].set_title('Dropout Rate by Debtor')

sns.barplot(x='Tuition_fees_up_to_date', y=(df['Status']=='Dropout').astype(int), data=df, ax=axes[0,2])
axes[0,2].set_title('Dropout Rate by Tuition Fees Status')

sns.barplot(x='Scholarship_holder', y=(df['Status']=='Dropout').astype(int), data=df, ax=axes[1,0])
axes[1,0].set_title('Dropout Rate by Scholarship')

sns.boxplot(x='Status', y='Age_at_enrollment', data=df, ax=axes[1,1])
axes[1,1].set_title('Age by Status')

sns.boxplot(x='Status', y='Curricular_units_1st_sem_grade', data=df, ax=axes[1,2])
axes[1,2].set_title('Semester 1 Grade by Status')

plt.tight_layout()
plt.show()

df_clean = df.copy()

# Encode target: Dropout, Enrolled, Graduate -> 0, 1, 2
le_target = LabelEncoder()
df_clean['Status'] = le_target.fit_transform(df_clean['Status'])

X = df_clean.drop(columns=['Status'])
y = df_clean['Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape, X_test_scaled.shape)

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_importance.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.show()

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le_target, 'label_encoder.joblib')

df.to_csv('dashboard_data.csv', index=False)