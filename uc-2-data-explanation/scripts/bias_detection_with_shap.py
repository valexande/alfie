import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
df = pd.read_csv("C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/alert-data-uc2-demographics.csv")

# Encode categorical variables
categorical_cols = ['gender', 'ethnicity', 'race']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(columns=['alert'])
y = df['alert']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plots
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)

# Save feature contributions by demographic features
demographic_features = ['gender', 'age', 'race']
demographic_shap_values = pd.DataFrame(shap_values[1], columns=X_test.columns)[demographic_features]

# Map encoded values back to original labels
df['gender_label'] = label_encoders['gender'].inverse_transform(df['gender'])
df['race_label'] = label_encoders['race'].inverse_transform(df['race'])
df['ethnicity_label'] = label_encoders['ethnicity'].inverse_transform(df['ethnicity'])

# Alert counts per demographic category with proper labels
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Number of Alerts per Demographic Category', fontsize=16)

df[df['alert'] == 1]['age'].value_counts().sort_index().plot(kind='bar', ax=axs[0, 0])
axs[0, 0].set_title('Alerts per Age')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('Number of Alerts')

df[df['alert'] == 1]['race_label'].value_counts().plot(kind='bar', ax=axs[0, 1])
axs[0, 1].set_title('Alerts per Race')
axs[0, 1].set_xlabel('Race')
axs[0, 1].set_ylabel('Number of Alerts')

df[df['alert'] == 1]['gender_label'].value_counts().plot(kind='bar', ax=axs[1, 0])
axs[1, 0].set_title('Alerts per Gender')
axs[1, 0].set_xlabel('Gender')
axs[1, 0].set_ylabel('Number of Alerts')

df[df['alert'] == 1]['ethnicity_label'].value_counts().plot(kind='bar', ax=axs[1, 1])
axs[1, 1].set_title('Alerts per Ethnicity')
axs[1, 1].set_xlabel('Ethnicity')
axs[1, 1].set_ylabel('Number of Alerts')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# -------------------
# Fairness Analysis
# -------------------

def group_metrics(df, predictions, group_col):
    df_copy = df.copy()
    df_copy['prediction'] = predictions
    df_copy['true'] = y.values
    groups = df_copy[group_col].unique()
    print(f"\nFairness Report by {group_col.capitalize()}:")
    for group in groups:
        subset = df_copy[df_copy[group_col] == group]
        acc = accuracy_score(subset['true'], subset['prediction'])
        prec = precision_score(subset['true'], subset['prediction'], zero_division=0)
        rec = recall_score(subset['true'], subset['prediction'], zero_division=0)
        print(f"  {group}: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}")

# Run fairness checks
predictions = model.predict(X)
group_metrics(df, predictions, 'gender')
group_metrics(df, predictions, 'race')
group_metrics(df, predictions, 'ethnicity')
