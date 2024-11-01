# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)

# Replace with your own secret key
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload and reports directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'reports'), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_recidivism(input_data):
    # For demonstration purposes, generate a random probability between 0 and 1
    recidivism_prob = random.uniform(0, 1)
    return recidivism_prob

def predict_violent_recidivism(input_data):
    # Generate a random probability between 0 and 1 for violent recidivism
    violent_recidivism_prob = random.uniform(0, 1)
    return violent_recidivism_prob

def generate_pdf_report(input_data, recidivism_prob, violent_recidivism_prob):
    pdf_filename = 'report.pdf'
    c = canvas.Canvas(os.path.join('static', 'reports', pdf_filename), pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2.0, height - 50, "Recidivism Prediction Report")

    # Input Data Section
    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, "Input Data:")
    y -= 20
    for key, value in input_data.items():
        c.drawString(70, y, f"{key.replace('_', ' ').title()}: {value}")
        y -= 20

    # Predictions
    y -= 10
    c.drawString(50, y, "Predictions:")
    y -= 20
    c.drawString(70, y, f"Recidivism Probability: {round(recidivism_prob * 100, 2)}%")
    y -= 20
    c.drawString(70, y, f"Violent Recidivism Probability: {round(violent_recidivism_prob * 100, 2)}%")

    # Save PDF
    c.save()
    return pdf_filename

def extract_data_from_pdf(file_path):
    # For demonstration purposes, return dummy data
    # In practice, use PyPDF2 or pdfminer.six to extract text from the PDF
    extracted_data = {
        'age': 35,
        'juv_fel_count': 0,
        'juv_misd_count': 1,
        'juv_other_count': 0,
        'priors_count': 2,
        'days_b_screening_arrest': -5,
        'c_days_from_compas': 0,
        'decile_score': 4,
        'score_text': 'Medium',
        'c_charge_degree': 'F',
        'sex': 'Male',
        'race': 'African-American'
    }
    return extracted_data

@app.context_processor
def inject_now():
    return {'current_year': datetime.utcnow().year}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_form', methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
        # Collect form data
        input_data = request.form.to_dict()
        # Convert numerical fields to appropriate types
        numerical_fields = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                            'days_b_screening_arrest', 'c_days_from_compas', 'decile_score']
        for field in numerical_fields:
            input_data[field] = float(input_data[field])
        # Generate predictions
        recidivism_prob = predict_recidivism(input_data)
        violent_recidivism_prob = predict_violent_recidivism(input_data)
        recidivism_score = 'High' if recidivism_prob > 0.5 else 'Low'
        violent_recidivism_score = 'High' if violent_recidivism_prob > 0.5 else 'Low'
        # Generate PDF report
        pdf_filename = generate_pdf_report(input_data, recidivism_prob, violent_recidivism_prob)
        return render_template('results.html',
                               recidivism_prob=recidivism_prob,
                               recidivism_score=recidivism_score,
                               violent_recidivism_prob=violent_recidivism_prob,
                               violent_recidivism_score=violent_recidivism_score,
                               pdf_filename=pdf_filename)
    else:
        prefill_data = {}
        return render_template('input_form.html', prefill_data=prefill_data)

@app.route('/scan_pdf', methods=['GET', 'POST'])
def scan_pdf():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Extract data from the PDF
            extracted_data = extract_data_from_pdf(file_path)
            # Remove the uploaded file after processing
            os.remove(file_path)
            # Render the input form with pre-filled data
            return render_template('input_form.html', prefill_data=extracted_data)
    else:
        return render_template('scan_pdf.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(os.path.join('static', 'reports'), filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)