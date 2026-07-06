import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting training job...")

# 1. Create the output directory where Azure looks for artifacts
os.makedirs('outputs', exist_ok=True)

# 2. Your Model logic (Dummy layout)
data = {"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8], "label": [0, 1, 0, 1]}
df = pd.DataFrame(data)
X = df[["feature1", "feature2"]]
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

# 3. Save the model into the outputs folder
joblib.dump(model, 'outputs/model.pkl')
print("Model trained and saved to outputs/model.pkl!")