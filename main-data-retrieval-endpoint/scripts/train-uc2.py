import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load your dataset
# -----------------------------
df = pd.read_csv("C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/alert-data-uc2-demographics.csv")  # Replace with your actual path

# -----------------------------
# 2. Encode categorical columns
# -----------------------------
categorical_cols = ['gender', 'ethnicity', 'race']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# 3. Define features and target
# -----------------------------
X = df.drop(columns=['alert'])
y = df['alert']

# -----------------------------
# 4. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate performance
# -----------------------------
print("Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# -----------------------------
# 7. Save trained model
# -----------------------------
joblib.dump(model, "../csv-pkl-json/model.pkl")
print("Model saved to model.pkl")

# Optional: Save the label encoders if needed later for decoding
joblib.dump(label_encoders, "../csv-pkl-json/label_encoders.pkl")
print("Label encoders saved to label_encoders.pkl")
