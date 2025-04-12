# 🕵️‍♂️ Twitter Fake Account Detector
Detects whether a Twitter account is fake or real using machine learning. Built with Python, Scikit-learn, and a sleek Bootstrap-powered frontend.

A machine learning-powered web app that identifies fake Twitter accounts using user profile features such as tweet count, follower ratio, bio length, language, and more.

## 🚀 Project Overview

This project uses a Random Forest Classifier to detect fake Twitter accounts. It analyzes publicly available Twitter user metadata and predicts whether a given profile is likely to be real or fake. The front-end is built using Bootstrap, and the back-end is developed in Python using Flask and Scikit-learn.

## 📊 Features

- Predicts whether a Twitter account is real or fake
- Probability and confidence score for each prediction
- Beautiful, dark-themed user interface
- Gender detection based on the user's first name
- Language encoding and normalization of profile stats

## 🧠 Tech Stack

- **Frontend**: HTML, CSS (Bootstrap), JavaScript (jQuery)
- **Backend**: Python, Flask
- **ML Model**: Random Forest Classifier (scikit-learn)
- **Other Libraries**: pandas, numpy, gender-guesser, matplotlib, joblib

## 📁 Dataset

Two datasets are used:
- `realusers.csv`
- `fakeusers.csv`

Make sure to place them in the `datasets/` folder.

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twitter-fake-account-detector.git
   cd twitter-fake-account-detector
   
Install dependencies:

pip install -r requirements.txt
Train the model (if needed) or use the pre-trained model:

python fake_account_detector.py
Run the web app:

flask run
Open your browser at:

http://127.0.0.1:5000/
📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙋‍♂️ Author
Udit Jhanjhariya
Manipal University Jaipur
B.Tech CSE — 2027

🌟 If you like this project, consider giving it a ⭐ on GitHub!
Copy
Edit

---

Let me know if you want it customized with your real name, GitHub username, or graduation year — I can auto-fill 
