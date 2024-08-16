import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render

from .classification import load_model, predict_single_point
from .forms import PossumPredictionForm, SymptomForm


def home(request):
    return render(request, 'home.html')

def classification(request):
    prediction = None
    confidence = None

    if request.method == 'POST':
        form = SymptomForm(request.POST)
        if form.is_valid():
            # Collect and process symptoms
            symptoms = [
                form.cleaned_data['systemic_illness'],
                form.cleaned_data['rectal_pain'],
                form.cleaned_data['sore_throat'],
                form.cleaned_data['penile_oedema'],
                form.cleaned_data['oral_lesions'],
                form.cleaned_data['solitary_lesion'],
                form.cleaned_data['swollen_tonsils'],
                form.cleaned_data['hiv_infection'],
                form.cleaned_data['sexually_transmitted_infection']
            ]

            # Check if all symptoms are negative
            all_negative = (
                symptoms[0] == 'None' and  # Systemic illness is 'None'
                not any(symptoms[1:])      # All boolean fields are False
            )
            
            if all_negative:
                # If all symptoms are negative, assume the prediction is negative
                prediction = "Negative"
                confidence = "High (Based on no symptoms)"
            else:
                # Map systemic illness to one-hot encoding
                illness_mapping = {
                    'None': [1, 0, 0, 0],
                    'Fever': [0, 1, 0, 0],
                    'Muscle Aches and Pain': [0, 0, 1, 0],
                    'Swollen Lymph Nodes': [0, 0, 0, 1]
                }
                illness_encoded = illness_mapping.get(symptoms[0], [1, 0, 0, 0])
                symptoms_encoded = illness_encoded + symptoms[1:]

                # Load all models
                model_names = ['naive_bayes', 'decision_tree', 'random_forest', 'svm', 'knn']
                models = {name: load_model(name) for name in model_names}

                # Get predictions from all models
                results = predict_single_point(models, symptoms_encoded)

                # Find the prediction with the highest confidence
                highest_confidence_model = max(results, key=lambda model: results[model][1])
                prediction, confidence = results[highest_confidence_model]
                model_name = highest_confidence_model

                # Hilariously, passing a 0.0 kills the prediction block in the template
                prediction = "Positive" if prediction == 1.0 else "Negative"

                # Print to debug
                print(f"Prediction: {prediction}, Confidence: {confidence}")

                # Render the result
                return render(request, 'classification.html', {
                    'form': form,
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_name': model_name,
                })
        else:
            # Print form errors to debug why validation failed
            print(form.errors)

    else:
        form = SymptomForm()

    return render(request, 'classification.html', {'form': form})

def regression(request):
    # Load the dataset
    df = pd.read_csv('static/possum.csv')

    # Select three random rows
    sample_data = df.sample(3).to_dict(orient='records')

    prediction = None
    confidence = None
    model_name = "Linear Regression"

    if request.method == 'POST':
        form = PossumPredictionForm(request.POST)
        if form.is_valid():
            pop_mapping = {'Vic': 0, 'other': 1}
            sex_mapping = {'m': 1, 'f': 0}
            
            # Collect and process form data
            features = np.array([
                int(form.cleaned_data['site']),
                pop_mapping[form.cleaned_data['Pop_other']],  # Map 'Vic'/'other' to 0/1
                sex_mapping[form.cleaned_data['sex_m']],  # Map 'm'/'f' to 1/0
                float(form.cleaned_data['age']),
                float(form.cleaned_data['skullw']),
                float(form.cleaned_data['totlngth']),
                float(form.cleaned_data['taill']),
                float(form.cleaned_data['footlgth']),
                float(form.cleaned_data['earconch']),
                float(form.cleaned_data['eye']),
                float(form.cleaned_data['chest']),
                float(form.cleaned_data['belly']),
            ]).reshape(1, -1)

            # Load the scaler
            scaler_path = f'{settings.BASE_DIR}/machine_learning/data/scaler_hd.pkl'
            scaler = joblib.load(scaler_path)
            features_scaled = scaler.transform(features)

            # Load and predict using all models
            model_names = ['random_forest', 'gradient_boosting', 'svr']
            predictions = {}
            for model_name in model_names:
                model_path = f'{settings.BASE_DIR}/machine_learning/data/{model_name}_hd_model.pkl'
                model = joblib.load(model_path)
                predictions[model_name] = model.predict(features_scaled)[0]

            # Prepare sample data to show in the table
            df = pd.read_csv(f'{settings.STATICFILES_DIRS[0]}/possum.csv')
            sample_data = df[['site', 'Pop', 'sex', 'age', 'hdlngth', 'skullw', 'totlngth', 'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly']].sample(3).to_dict(orient='records')

            # Render the result
            return render(request, 'regression.html', {
                'form': form,
                'predictions': predictions,
                'sample_data': sample_data,
            })
    else:
        form = PossumPredictionForm()

    # Always return an HttpResponse, even if the method is GET or the form is not valid
    return render(request, 'regression.html', {
        'form': form,
        'sample_data': sample_data,
    })