from flask_wtf import FlaskForm
from wtforms import IntegerField, StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, NumberRange

class InputForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=0)])
    sex = SelectField('Sex', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    priors_count = IntegerField('Prior Offenses', validators=[DataRequired(), NumberRange(min=0)])
    race = SelectField('Race', choices=[
        ('Caucasian', 'Caucasian'), 
        ('African-American', 'African-American'), 
        ('Hispanic', 'Hispanic'), 
        ('Other', 'Other'), 
        ('Asian', 'Asian'), 
        ('Native American', 'Native American')
    ], validators=[DataRequired()])
    juv_fel_count = IntegerField('Juvenile Felony Count', validators=[DataRequired(), NumberRange(min=0)])
    juv_misd_count = IntegerField('Juvenile Misdemeanor Count', validators=[DataRequired(), NumberRange(min=0)])
    juv_other_count = IntegerField('Juvenile Other Count', validators=[DataRequired(), NumberRange(min=0)])
    days_b_screening_arrest = IntegerField('Days Between Screening and Arrest', validators=[DataRequired(), NumberRange(min=0)])
    c_days_from_compas = IntegerField('Days from COMPAS', validators=[DataRequired(), NumberRange(min=0)])
    c_charge_degree = SelectField('Charge Degree', choices=[
        ('(F1)', 'First Degree Felony'), 
        ('(F2)', 'Second Degree Felony'),
        ('(F3)', 'Third Degree Felony'),
        ('(F6)', 'Sixth Degree Felony'),
        ('(M1)', 'First Degree Misdemeanor'),
        ('(M2)', 'Second Degree Misdemeanor')
    ], validators=[DataRequired()])
    decile_score = IntegerField('Decile Score', validators=[DataRequired(), NumberRange(min=0, max=10)])
    score_text = SelectField('Risk Level', choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], validators=[DataRequired()])
    submit = SubmitField('Submit')
