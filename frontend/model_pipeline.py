from models.recidivism.classification.model_pipeline import CustomPipeline as RecidivismPipeline
# This file interacts with the ML pipeline. 

# For example:
def real_predict(input_data):
    pipeline = RecidivismPipeline()

    recidivism_pred, recidivism_prob, violence_pred, violence_prob = pipeline.real_predict(input_data)
    
    return recidivism_pred, recidivism_prob, violence_pred, violence_prob
