import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import data.recidivism.data as recidivism_data
import data.recidivism.compasspreproccessing as recidivism_compass_preproccessing  # Import your COMPAS preprocessor
import data.violence.compasspreproccessing as violence_compass_preproccessing  # Import your COMPAS preprocessor
import data.recidivism.preproccessing as recidivism_preproccessing
import data.violence.preproccessing as violence_preproccessing
from models.recidivism.classification.model_pipeline import CustomPipeline
from models.recidivism.date.model_pipeline import CustomPipeline as DatePipeline
from models.recidivism.severity.model_pipeline import CustomPipeline as SeverityPipeline
import evaluations.racialBiasDetection as racialBiasDetection
import postprocessing.classification.equalized_odds as equalized_odds
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

def preprocess_data(input_data, prediction_type):
    # Ensure input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])  # Convert dictionary to DataFrame
    print(input_data)
    if prediction_type.lower() == "recidivism":
        # Remove violence-related fields
        keys_to_remove = ['v_decile_score', 'v_score_text']
        input_data = input_data.drop(columns=[key for key in keys_to_remove if key in input_data.columns], errors='ignore')
        # Preprocess using your recidivism preprocessing function
        processed_data = recidivism_preproccessing.preproccess(input_data)
    elif prediction_type.lower() == "violence":
        # Remove recidivism-related fields
        keys_to_remove = ['decile_score', 'score_text']
        input_data = input_data.drop(columns=[key for key in keys_to_remove if key in input_data.columns], errors='ignore')
        # Preprocess using your violence preprocessing function
        processed_data = violence_preproccessing.preproccess(input_data)
    else:
        raise ValueError("Invalid prediction type. Please specify 'violence' or 'recidivism'.")

    # Ensure the output is a DataFrame
    if not isinstance(processed_data, pd.DataFrame):
        processed_data = pd.DataFrame(processed_data)
    return processed_data

def predict(input_data, prediction_type):
    # First preprocess the data obviously
    processed_data = preprocess_data(input_data, prediction_type)
    model = CustomPipeline(section_equalizer="race", adversarial=True)
    preloadName = None
    # Making sure to specify the correct model and reference type
    if (prediction_type.lower() == "violence"):
        preloadName = "adversarialNoPostViolenceClassification"
    elif (prediction_type.lower() == "recidivism"):
        preloadName = "adversarialNoPostRecidivismClassification"
    else:
        raise ValueError("Invalid prediction type. Please specify 'violence' or 'recidivism'.")
    
    predictions = model.real_predict(processed_data, preloadName = preloadName)
    return predictions

def predict_date(input_data, prediction_type):
    # First preprocess the data obviously
    processed_data = preprocess_data(input_data, prediction_type)
    model = DatePipeline(section_equalizer="race", adversarial=True)
    preloadName = None
    # Making sure to specify the correct model and reference type
    if (prediction_type.lower() == "violence"):
        preloadName = "adversarialNoPostViolenceDate"
    elif (prediction_type.lower() == "recidivism"):
        preloadName = "adversarialNoPostRecidivismDate"
    else:
        raise ValueError("Invalid prediction type. Please specify 'violence' or 'recidivism'.")
    
    predictions = model.real_predict(processed_data, preloadName = preloadName)
    return predictions

def predict_severity(input_data, recidivism_verdict, violence_verdict):
    # First preprocess the data obviously
    recidivism_processed_data = preprocess_data(input_data, 'recidivism')
    violence_processed_data = preprocess_data(input_data, 'violence')

    model = SeverityPipeline(section_equalizer="race", adversarial=True)
    
    predictions = model.real_predict(recidivism_processed_data, violence_processed_data, recidivism_verdict = recidivism_verdict, violence_verdict = violence_verdict, 
                                     recidivismPreloadName = "adversarialNoPostRecidivismSeverity", violencePreloadName = "adversarialNoPostViolenceSeverity")
    return predictions