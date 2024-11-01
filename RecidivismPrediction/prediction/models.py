from django.db import models

class PredictionHistory(models.Model):
    input_data = models.TextField()
    result = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction on {self.created_at}"
