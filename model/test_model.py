import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import gender_guesser.detector as gender
import csv
import re

print("🔹 Loading model...")
model = load('twitter_fake_account_detector.joblib')

print("🔹 Loading datasets...")
real_users = pd.read_csv('realusers.csv')
fake_users = pd.read_csv('fakeusers.csv')

print(f"✅ Loaded {len(real_users)} real and {len(fake_users)} fake users")

# -------------------------------------------------
# Combine both datasets
# -------------------------------------------------
X = pd.concat([real_users, fake_users], ignore_index=True)
y_true = len(fake_users)*[0] + len(real_users)*[1]

# -------------------------------------------------
# Preprocessing (same as model.py)
# -------------------------------------------------
print("🧩 Preprocessing data (gender + language encoding)...")

sex_predictor = gender.Detector(case_sensitive=False)
X['First Name'] = X['name'].astype(str).str.split(' ').str.get(0)

def clean_name(name):
    return re.sub(r'[^\x00-\x7F]+', '', str(name))

X['First Name'] = X['First Name'].apply(clean_name)
X['Predicted Sex'] = X['First Name'].apply(sex_predictor.get_gender)
sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
X['Predicted Sex'] = X['Predicted Sex'].apply(lambda x: 'unknown' if x == 'andy' else x)
X['Sex Code'] = X['Predicted Sex'].map(sex_dict).fillna(0).astype(int)

lang_list = list(enumerate(np.unique(X['lang'].astype(str))))
lang_dict = {name: i for i, name in lang_list}
X['lang_code'] = X['lang'].map(lang_dict).fillna(0).astype(int)

# Feature selection
features = ['Sex Code','statuses_count','followers_count','friends_count','favourites_count','listed_count','lang_code']
X = X[features]

# -------------------------------------------------
# Prediction
# -------------------------------------------------
print("🧠 Making predictions...")
y_pred = model.predict(X)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
print("\n📊 Calculating performance metrics...")
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1-Score:  {f1:.4f}")

results_text = f"""
Twitter Fake Account Detector - Testing Results
-----------------------------------------------
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}

Classification Report:
{report}
"""

with open('testing_results.txt', 'w') as f:
    f.write(results_text)
print("📝 Results saved to testing_results.txt")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix - Twitter Fake Account Detector')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("📊 Confusion matrix saved as confusion_matrix.png")

# Save testing history
with open('testing_history.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([accuracy, precision, recall, f1])

print("📈 Appended this run to testing_history.csv")
print("\n✅ Testing completed successfully!")
