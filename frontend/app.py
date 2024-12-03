# app.py

import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file
from werkzeug.utils import secure_filename
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

import main

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

def predict_recidivism_classification(input_data):
    # For demonstration purposes, generate a random probability between 0 and 1
    recidivism_prediction, recidivism_probability = main.predict(input_data, "recidivism")
    return recidivism_probability

def predict_violence_classification(input_data):
    # For demonstration purposes, generate a random probability between 0 and 1
    violence_prediction, violence_probability = main.predict(input_data, "violence")
    return violence_probability

def predict_recidivism_date(input_data, prediction_type):
    # First preprocess the data obviously
    recidivism_date = main.predict_date(input_data, prediction_type)
    return recidivism_date

def predict_violence_date(input_data, prediction_type):
    # First preprocess the data obviously
    violence_date = main.predict_date(input_data, prediction_type)
    return violence_date

def predict_severity(input_data, recidivism_verdict, violence_verdict):
    # First preprocess the data obviously
    recidivism_severity = main.predict_severity(input_data, recidivism_verdict, violence_verdict)
    return recidivism_severity

def generate_pdf_report(
    user_data,
    recidivism_prob,
    recidivism_score,
    violent_recidivism_prob,
    violent_recidivism_score,
    recidivism_date,
    violence_date,
    recid_severity_probability,
    recid_severity_classification,
    violence_severity_probability,
    violence_severity_classification
):
    # Set the file name
    pdf_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join("frontend", "static", "reports", pdf_filename)

    # Create a SimpleDocTemplate
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []

    # Add Logo and Header
    logo_path = os.path.join("frontend", "static", "images", "logo.png")  # Update with your logo path
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=100, height=50)
        elements.append(logo)
    elements.append(Spacer(1, 20))

    header = Paragraph("<strong>Recidivism and Violent Crime Risk Report</strong>", getSampleStyleSheet()["Title"])
    elements.append(header)

    date = Paragraph(f"<i>Date: {datetime.now().strftime('%B %d, %Y')}</i>", getSampleStyleSheet()["Normal"])
    elements.append(date)
    elements.append(Spacer(1, 20))

    # Add User Data
    user_data_section = Paragraph("<strong>User Information</strong>", getSampleStyleSheet()["Heading2"])
    elements.append(user_data_section)
    user_data_table = Table([[key, value] for key, value in user_data.items()])
    user_data_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(user_data_table)
    elements.append(Spacer(1, 20))

    # Summary Section
    summary_section = Paragraph("<strong>Summary</strong>", getSampleStyleSheet()["Heading2"])
    elements.append(summary_section)

    summary_data = [
        ["Recidivism Probability", f"{round(recidivism_prob[0][0] * 100, 2)}%"],
        ["Recidivism Risk Level", recidivism_score],
        ["Violent Recidivism Probability", f"{round(violent_recidivism_prob[0][0] * 100, 2)}%"],
        ["Violent Recidivism Risk Level", violent_recidivism_score],
        ["Predicted Recidivism Date", recidivism_date],
        ["Predicted Violent Crime Date", violence_date],
    ]
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Severity Section
    severity_section = Paragraph("<strong>Severity Predictions</strong>", getSampleStyleSheet()["Heading2"])
    elements.append(severity_section)

    severity_data = [
        ["Recidivism Severity Probability", f"{round(recid_severity_probability * 100, 2)}%" if recid_severity_probability else "N/A"],
        ["Recidivism Severity Classification", recid_severity_classification if recid_severity_classification else "N/A"],
        ["Violent Crime Severity Probability", f"{round(violence_severity_probability * 100, 2)}%" if violence_severity_probability else "N/A"],
        ["Violent Crime Severity Classification", violence_severity_classification if violence_severity_classification else "N/A"],
    ]
    severity_table = Table(severity_data)
    severity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(severity_table)
    elements.append(Spacer(1, 20))

    # Footer
    footer = Paragraph(
        "<i>Generated by Fighting Against Racial Bias in the Criminal Justice System</i>",
        ParagraphStyle(name="Footer", fontSize=10, alignment=1)
    )
    elements.append(Spacer(1, 50))
    elements.append(footer)

    # Build the PDF
    doc.build(elements)

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
        print(input_data)
        # Convert numerical fields to appropriate types
        numerical_fields = [
            'age', 'juv_fel_count', 'juv_misd_count', 
            'juv_other_count', 'priors_count', 
            'days_b_screening_arrest', 'c_days_from_compas', 
            'decile_score', 'v_decile_score'
        ]

        for field in numerical_fields:
            input_data[field] = float(input_data[field])

        for field in numerical_fields:
            try:
                input_data[field] = float(input_data[field])  # Convert to float
            except ValueError as e:
                raise ValueError(f"Field '{field}' should be numeric but got: {input_data[field]}") from e
        # Generate predictions
        recidivism_prob = predict_recidivism_classification(input_data)
        violent_recidivism_prob = predict_violence_classification(input_data)
        recidivism_score = 'High' if recidivism_prob > 0.5 else 'Low'
        violent_recidivism_score = 'High' if violent_recidivism_prob > 0.5 else 'Low'
        recidivism_date = predict_recidivism_date(input_data, "recidivism")[0][0]
        violence_date = predict_violence_date(input_data, "violence")[0][0]
        severity_mapping = {
            0: "Misdemeanor",
            1: "M1 (1st Degree Misdemeanor)",
            2: "M2 (2nd Degree Misdemeanor)",
            3: "F1 (1st Degree Felony)",
            4: "F2 (2nd Degree Felony)",
            5: "F3 (3rd Degree Felony)",
            6: "F4 (4th Degree Felony)",
            7: "F5 (5th Degree Felony)",
            8: "F6 (6th Degree Felony)",
            9: "F7 (7th Degree Felony)",
            10: "F8 (8th Degree Felony)",
        }
        recidivism_verdict = 1 if recidivism_prob > 0.5 else 0
        violence_verdict = 1 if violent_recidivism_prob > 0.5 else 0
        if (recidivism_verdict == 1 and violence_verdict == 1):
            recid_severity_probability, recid_severity_classification, violence_severity_probability, violence_severity_classification = predict_severity(input_data, recidivism_verdict, violence_verdict)
            recid_severity_classification = severity_mapping.get(recid_severity_classification[0], "Unknown")
            violence_severity_classification = severity_mapping.get(violence_severity_classification[0], "Unknown")
            recid_severity_probability = recid_severity_probability[0][0]
            violence_severity_probability = violence_severity_probability[0][0]
        elif recidivism_verdict == 1:
            recid_severity_probability, recid_severity_classification = predict_severity(input_data, recidivism_verdict, violence_verdict)
            violence_severity_probability, violence_severity_classification = None, None
            recid_severity_classification = severity_mapping.get(recid_severity_classification[0], "Unknown")
            recid_severity_probability = recid_severity_probability[0][0]
        elif violence_verdict == 1:
            violence_severity_probability, violence_severity_classification = predict_severity(input_data, recidivism_verdict, violence_verdict)
            recid_severity_probability, recid_severity_classification = None, None
            violence_severity_classification = severity_mapping.get(violence_severity_classification[0], "Unknown")
            violence_severity_probability = violence_severity_probability[0][0]
        else:
            recid_severity_probability, recid_severity_classification, violence_severity_probability, violence_severity_classification = None, None, None, None
        # Generate PDF report

        pdf_filename = generate_pdf_report(input_data, recidivism_prob, recidivism_score, violent_recidivism_prob, violent_recidivism_score, 
                                           recidivism_date, violence_date, recid_severity_probability, recid_severity_classification, violence_severity_probability, violence_severity_classification)
        return render_template('results.html',
                               recidivism_prob=recidivism_prob,
                               recidivism_score=recidivism_score,
                               violent_recidivism_prob=violent_recidivism_prob,
                               violent_recidivism_score=violent_recidivism_score,
                               recidivism_date=recidivism_date,
                               violence_date=violence_date,
                                recid_severity_probability=recid_severity_probability,
                                recid_severity_classification=recid_severity_classification,
                                violence_severity_probability=violence_severity_probability,
                                violence_severity_classification=violence_severity_classification,
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