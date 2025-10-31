# 🕵️‍♂️ Twitter Fake Account Detector

A machine learning-powered web application that helps identify potentially fake Twitter profiles using various profile features and characteristics. Built with Python, Scikit-learn, and a sleek Bootstrap-powered frontend.

## 🚀 Features

- Real-time prediction of Twitter profile authenticity
- Analysis of multiple profile features including:
  - Tweet count
  - Follower count
  - Following count
  - Favorites count
  - Listed count
  - Account age
  - Bio length
  - Language preference
  - Gender detection
- Beautiful, dark-themed user interface
- Detailed prediction confidence scores
- Gender detection based on the user's first name
- Language encoding and normalization of profile stats

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (Bootstrap), JavaScript (jQuery)
- **Backend**: Python, Flask
- **ML Model**: Random Forest Classifier (scikit-learn)
- **Other Libraries**: 
  - pandas
  - numpy
  - gender-guesser
  - matplotlib
  - joblib
  - waitress

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Twitter-Fake-Profile-Detection.git
cd Twitter-Fake-Profile-Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the Flask application:
```bash
cd model
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8080
```

3. Enter the Twitter profile details in the form and click "Predict" to get the analysis.

## 📊 Model Details

The application uses a Random Forest Classifier trained on various Twitter profile features to predict the authenticity of profiles. The model considers:

- Account activity metrics
- Social engagement ratios
- Profile completeness
- Account age and growth patterns
- Language and demographic features

## 📁 Project Structure

```
Twitter-Fake-Profile-Detection/
├── model/
│   ├── app.py                 # Flask application
│   ├── model.py               # Model training code
│   ├── twitter_fake_account_detector.joblib  # Trained model
│   ├── templates/             # HTML templates
│   └── datasets/              # Training datasets
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 📚 Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- joblib
- gender-guesser
- waitress
- matplotlib
- numpy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙋‍♂️ Author

**Udit Jhanjhariya**  
Manipal University Jaipur  
B.Tech CSE — 2027

## 🙏 Acknowledgments

- Dataset used for training
- Contributors and maintainers
- Open-source libraries used in the project


If you like this project, consider giving it a ⭐ on GitHub!
