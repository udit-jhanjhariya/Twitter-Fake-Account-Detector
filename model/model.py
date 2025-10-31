#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import re
import gender_guesser.detector as gender
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

try:

    
    print("Loading datasets...")
    # Load datasets
    real_users = pd.read_csv('datasets/realusers.csv')
    fake_users = pd.read_csv('datasets/fakeusers.csv')

    print(f"Loaded {len(real_users)} real users and {len(fake_users)} fake users")
    print(f"Real users shape: {real_users.shape}")
    print(f"Fake users shape: {fake_users.shape}")

    # Concatenate both datasets
    print("\nConcatenating datasets...")
    X = pd.concat([real_users, fake_users])
    print(f"Combined dataset shape: {X.shape}")

    # Check for missing values
    print('\nChecking for missing values...')
    columns_to_check = ['Unnamed: 0', 'id', 'name', 'screen_name', 'statuses_count',
           'followers_count', 'friends_count', 'favourites_count', 'listed_count',
           'created_at', 'url', 'lang', 'time_zone', 'location', 'default_profile',
           'default_profile_image', 'geo_enabled', 'profile_image_url',
           'profile_banner_url', 'profile_use_background_image',
           'profile_background_image_url_https', 'profile_text_color',
           'profile_image_url_https', 'profile_sidebar_border_color',
           'profile_background_tile', 'profile_sidebar_fill_color',
           'profile_background_image_url', 'profile_background_color',
           'profile_link_color', 'utc_offset', 'protected', 'verified',
           'description', 'updated', 'dataset', 'age_in_days',
           'ratio statuses_count/age', 'ratio Favorites/age',
           'ratio Friends/Followers', 'length_of_bio', 'reputation']

    columns_to_remove = ['Unnamed: 0']

    for column in columns_to_check:
        try:
            missing_count = X[column].isnull().sum()
            print(f"{column}: {missing_count} missing values")

            # Check if missing count is greater than 2000
            if missing_count > 2000:
                columns_to_remove.append(column)
                print(f"  -> Removing {column} due to more than 2000 missing values")
        except KeyError:
            print(f"{column}: Column not found in the DataFrame")

    # Remove columns with more than 2000 null values
    print("\nRemoving columns with too many missing values...")
    X = X.drop(columns=columns_to_remove, axis=1)
    print(f"Remaining columns: {len(X.columns)}")

    print("\nSample of processed data:")
    print(X.head())

    # Assigning False '0' to fake_users list and true '1' to real_users list
    print("\nCreating target labels...")
    y = len(fake_users)*[0]+len(real_users)*[1]
    print(f"Created {len(y)} labels")

    print("\nProcessing gender information...")
    # Create a detector instance
    sex_predictor = gender.Detector(case_sensitive=False)

    # Extract the first name and predict their genders
    X['First Name'] = X['name'].str.split(' ').str.get(0)

    # To handle names that have unrecognized characters
    def clean_name(name):
        cleaned_name = re.sub(r'[^\x00-\x7F]+', '', name)
        return cleaned_name

    # Clean the 'First Name' values
    X['First Name'] = X['First Name'].apply(clean_name)
    X['Predicted Sex'] = X['First Name'].apply(sex_predictor.get_gender)

    # Mapping of Gender
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}

    # Handle 'unknown' values
    X['Predicted Sex'] = X['Predicted Sex'].apply(lambda x: 'unknown' if x == 'andy' else x)

    # Map the predicted genders to codes
    X['Sex Code'] = X['Predicted Sex'].map(sex_dict).astype(int)

    print("\nProcessing language information...")
    # Create a mapping of unique 'lang' values to codes
    lang_list = list(enumerate(np.unique(X['lang'])))
    lang_dict = {name: i for i, name in lang_list}

    print("Language mapping:")
    print(lang_dict)

    # Map 'lang' values to 'lang_code' and convert to integers
    X['lang_code'] = X['lang'].map(lambda lang: lang_dict[lang]).astype(int)

    # Feature Extraction from columns
    print("\nExtracting final features...")
    feature_columns_to_use = ['Sex Code','statuses_count','followers_count','friends_count','favourites_count','listed_count','lang_code']
    X = X[feature_columns_to_use]
    print(f"Final feature set shape: {X.shape}")

    print("\nTraining model...")
    from sklearn import impute, model_selection, metrics, preprocessing
    from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, make_scorer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train Random Forest model
    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    print("\nMaking predictions...")
    y_pred = rf_model.predict(X_test)

    # Calculate and print results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save results to a file
    print("\nSaving results...")
    with open('model_results.txt', 'w') as f:
        f.write("Twitter Fake Profile Detection - Model Results\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"Number of trees: 100\n")
        f.write(f"Training set size: {len(X_train)}\n")
        f.write(f"Test set size: {len(X_test)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save the model
    print("\nSaving model...")
    import joblib
    model_filename = 'twitter_fake_account_detector.joblib'
    joblib.dump(rf_model, model_filename)
    
    print(f"\nResults saved to 'model_results.txt'")
    print(f"Model saved as '{model_filename}'")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("Stack trace:")
    import traceback
    traceback.print_exc()


