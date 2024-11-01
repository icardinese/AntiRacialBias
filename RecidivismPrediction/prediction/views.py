from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputForm, UploadFileForm
from .models import PredictionHistory
from .utils import process_input, generate_report_pdf, generate_prediction
import json

# Home page view
def home(request):
    return render(request, 'prediction/home.html')

# Manual input view
def manual_input(request):
    if request.method == 'POST':
        form = InputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            result = generate_prediction(data)  # Call your ML model prediction
            report_pdf = generate_report_pdf(result)
            return HttpResponse(report_pdf, content_type='application/pdf')
    else:
        form = InputForm()
    return render(request, 'prediction/manual_input.html', {'form': form})

# Document scan view
def document_scan(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            # Extract data from the uploaded file (using OCR or another method)
            extracted_data = process_input(uploaded_file)
            result = generate_prediction(extracted_data)
            report_pdf = generate_report_pdf(result)
            return HttpResponse(report_pdf, content_type='application/pdf')
    else:
        form = UploadFileForm()
    return render(request, 'prediction/document_scan.html', {'form': form})

# Data transparency view (display comparative graphs)
def data_transparency(request):
    # Load COMPAS and your model's data for comparison
    compas_data = json.load(open('compas_data.json'))
    custom_model_data = json.load(open('custom_model_data.json'))
    return render(request, 'prediction/data_transparency.html', {
        'compas_data': compas_data,
        'custom_model_data': custom_model_data
    })

# Data history view
def data_history(request):
    history = PredictionHistory.objects.all()
    return render(request, 'prediction/data_history.html', {'history': history})
