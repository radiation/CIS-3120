import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def generate_images(df):
    image_base_path = '../static/'
    # Apparently dataframes are passed by reference
    df_copy = df.copy()

    # Convert boolean columns to integers
    bool_cols = df_copy.columns[df_copy.dtypes == 'bool'].tolist()
    df_copy[bool_cols] = df_copy[bool_cols].astype(int)

    # Compute correlations
    correlation_matrix = df_copy[bool_cols + ['MonkeyPox']].apply(
        lambda x: x.map({'Positive': 1, 'Negative': 0}) if x.name == 'MonkeyPox' else x
    ).corr()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Symptoms and MonkeyPox Result')
    plt.savefig(f'{image_base_path}/class_correlation_heatmap.png')

    fig, axes = plt.subplots(3, 3, figsize=(15, 12)) # Adjust the layout size based on the number of symptoms
    bool_cols = [col for col in df.columns if df[col].dropna().isin([True, False]).all()]
    fig.suptitle('Symptoms Count Based on MonkeyPox Result', fontsize=16)
    for i, col in enumerate(bool_cols):
        row, col_index = divmod(i, 3)
        sns.countplot(x=col, hue='MonkeyPox', data=df, ax=axes[row, col_index], palette='coolwarm')
        axes[row, col_index].set_title(f'{col} vs MonkeyPox')
        axes[row, col_index].set_xlabel('')
        axes[row, col_index].set_ylabel('Count')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{image_base_path}/class_count_plots.png')

    sns.set(style="whitegrid")

    # Bar Chart for the frequency of monkeypox cases
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(x='MonkeyPox', data=df, palette='coolwarm')
    ax.set_title('Frequency of Monkeypox Cases')
    ax.set_xlabel('MonkeyPox Result')
    ax.set_ylabel('Count')
    plt.savefig(f'{image_base_path}/class_bar_chart.png')

def generate_metrics_graph(models, X_test_scaled, y_test):
    # Metrics to collect
    metrics_summary = {
        'Model': [], 
        'Accuracy': [], 
        'Precision_weighted': [], 
        'Recall_weighted': [], 
        'F1_score_weighted': []
    }

    for name, model in models.items():
        # Predict on test data
        y_pred = model.predict(X_test_scaled)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Collecting data
        metrics_summary['Model'].append(name)
        metrics_summary['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics_summary['Precision_weighted'].append(report['weighted avg']['precision'])
        metrics_summary['Recall_weighted'].append(report['weighted avg']['recall'])
        metrics_summary['F1_score_weighted'].append(report['weighted avg']['f1-score'])

    # Creating the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = range(len(metrics_summary['Model']))  # the label locations
    width = 0.2  # the width of the bars

    # Plot each metric
    ax.bar(x, metrics_summary['Accuracy'], width, label='Accuracy', align='center')
    ax.bar([p + width for p in x], metrics_summary['Precision_weighted'], width, label='Precision (Weighted Avg)', align='center')
    ax.bar([p + 2*width for p in x], metrics_summary['Recall_weighted'], width, label='Recall (Weighted Avg)', align='center')
    ax.bar([p + 3*width for p in x], metrics_summary['F1_score_weighted'], width, label='F1-score (Weighted Avg)', align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks([p + 1.5*width for p in x])
    ax.set_xticklabels(metrics_summary['Model'])
    ax.legend()

    # Rotate the tick labels for better readability
    plt.xticks(rotation=45)
    plt.savefig('../static/class_metrics.png')

def load_model(model_name):
    # Construct the full path to the model file
    model_path = os.path.join(settings.BASE_DIR, 'machine_learning', 'data', f'{model_name}_model.pkl')
    
    # Load the model from the file
    model = joblib.load(model_path)
    
    return model

def predict_single_point(models, symptoms):
    predictions = {}
    
    for name, model in models.items():
        # Reshape symptoms for prediction
        symptoms_reshaped = [symptoms]  # Model expects a 2D array-like input
        print("Symptoms Reshaped:", symptoms_reshaped)
    
        # Get the prediction and confidence level
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(symptoms_reshaped)
            confidence = max(probabilities[0])
            prediction = model.predict(symptoms_reshaped)[0]
        else:
            # For models without `predict_proba`, we only return the prediction
            prediction = model.predict(symptoms_reshaped)[0]
            confidence = None  # Confidence not available
        
        predictions[name] = (prediction, confidence)
    
    print(predictions.items)
    return predictions

def main():
    # Ask the user if they want to build new models
    build_models = input("Do you want to build new models? (yes/no): ").strip().lower()

    # Read a CSV into a DataFrame
    df = pd.read_csv('../static/monkeypox.csv')

    # Generate a few visualizations (correlation matrix, bar graph, stacked bar chart)
    generate_images(df)

    # Define X (features) and y (target)
    systemic_illness_dummies = pd.get_dummies(df['Systemic Illness'], prefix='Illness')

    # Creating a new row with 'None' as Systemic Illness
    none_row = pd.DataFrame({
        'Systemic Illness': ['None'],
        'Rectal Pain': [False],
        'Sore Throat': [False],
        'Penile Oedema': [False],
        'Oral Lesions': [False],
        'Solitary Lesion': [False],
        'Swollen Tonsils': [False],
        'HIV Infection': [False],
        'Sexually Transmitted Infection': [False]
    })

    # Append the 'None' row using pd.concat
    df = pd.concat([df, none_row], ignore_index=True)

    # One-Hot Encode 'Systemic Illness'
    df_encoded = pd.get_dummies(df, columns=['Systemic Illness'])

    # Define the feature columns without duplicates
    bool_cols = df_encoded.columns[df_encoded.dtypes == 'bool'].tolist()
    X = df_encoded[bool_cols]  # All the boolean columns as features
    y = df['MonkeyPox'].map({'Positive': 1, 'Negative': 0}).dropna()

    X = X.loc[y.index]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Class Distribution in Training Data:")
    print(y_train.value_counts())

    # Standardizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    # Directory to save/load models
    save_directory = 'data/'

    if build_models == 'yes':
        # Apply SMOTE to balance the classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Fit models, perform 5-fold cross-validation, and save the models
        cross_val_results = {}
        for name, model in models.items():
            # Fitting the model
            model.fit(X_resampled, y_resampled)
            
            # Performing 5-fold cross-validation
            scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            cross_val_results[name] = scores
            print(f"{name} 5-Fold CV Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
            
            # Saving the model to a pickle file
            model_filename = f'{save_directory}{name.lower().replace(" ", "_")}_model.pkl'
            joblib.dump(model, model_filename)
            print(f"Model saved as {model_filename}")

        # Generate the accuracy plot
        plt.figure(figsize=(10, 6))
        model_names = list(cross_val_results.keys())
        mean_scores = [scores.mean() for scores in cross_val_results.values()]
        std_dev = [scores.std() for scores in cross_val_results.values()]

        plt.bar(model_names, mean_scores, yerr=std_dev, capsize=5, color='skyblue')
        plt.ylabel('Mean CV Accuracy')
        plt.title('5-Fold Cross-Validation Accuracy by Model')
        plt.savefig('images/plot.png')
    else:
        # Load the models from the pickle files
        for name in models.keys():
            model_filename = f'{save_directory}{name.lower().replace(" ", "_")}_model.pkl'
            models[name] = joblib.load(model_filename)
            print(f"Loaded {name} model from {model_filename}")

    # Predictions and classification reports
    predictions = {}
    classification_reports = {}
    accuracies = {}

    for name, model in models.items():
        # Making predictions
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        
        # Generating classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        classification_reports[name] = report
        
        # Collecting accuracy
        accuracies[name] = accuracy_score(y_test, y_pred)
        print(f"\n{name} Accuracy: {accuracies[name]:.2f}")

        test_cases = [
            # All symptoms are False, should ideally be Negative
            [1, 0, 0, 0, False, False, False, False, False, False, False, False],
            
            # Some symptoms are True, expect either Positive or Negative
            [0, 1, 0, 0, True, True, True, False, False, False, False, False],
            
            # All symptoms are True, expect Positive
            [0, 0, 0, 1, True, True, True, True, True, True, True, True]
        ]

        for i, case in enumerate(test_cases):
            prediction = model.predict([case])[0]
            confidence = model.predict_proba([case]).max()
            print(f"Prediction {i}: {True if prediction else False}, Confidence: {round(int(confidence),2)}")

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"{name} Feature Importances:")
            print(model.feature_importances_)

    # Simplified plot without detailed metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(models.keys()), [accuracies[model] for model in models.keys()], color='lightblue')
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    plt.savefig('images/accuracy.png')

    generate_metrics_graph(models, X_test_scaled, y_test)

if __name__ == "__main__":
    main()