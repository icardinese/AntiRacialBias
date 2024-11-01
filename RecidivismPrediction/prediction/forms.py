from django import forms

class InputForm(forms.Form):
    age = forms.IntegerField(label='Age', required=True, min_value = 0)
    sex = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')], required=True)
    priors_count = forms.IntegerField(label='Prior Offenses', required=True, min_value = 0)
    race = forms.ChoiceField(choices=[('Caucasian', 'Caucasian'), ('African-American', 'African-American'), ('Hispanic', 'Hispanic'), 
                                      ('Other', 'Other'), ('Asian', 'Asian'),
                                      ('Native American', 'Native American')], required=True)
    juv_fel_count = forms.IntegerField(label='Juvenile Felony Count', required=True, min_value = 0)
    juv_misd_count = forms.IntegerField(label='Juvenile Misdemeanor Count', required=True, min_value = 0)
    juv_other_count = forms.IntegerField(label='Juvenile Other Count', required=True, min_value = 0)
    days_b_screening_arrest = forms.IntegerField(label='Days between Screening and Arrest', required=True, min_value = 0)
    c_days_from_compas = forms.IntegerField(label='Days from COMPAS', required=True, min_value = 0)
    c_charge_degree = forms.ChoiceField(choices=[('(F1)', '(First degree Felony)'), ('(F2)', '(Second degree Felony)'), ('(F3)', '(Third degree Felony)'), ('(F6)', '(Sixth degree Felony)')
                    , ('(F7)', '(Seventh degree Felony)'), ('(M1)', '(First degree Misdemeanor)'), ('(M2)', '(Second degree Misdemeanor)')], required=True)
    decile_score = forms.IntegerField(label='Decile Score', required=True, min_value = 0, max_value = 10)
    score_text = forms.ChoiceField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], required=True)
    v_score_text = forms.ChoiceField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], required=True)
    v_decile_score = forms.IntegerField(label='Violence Decile Score', required=True, min_value = 0, max_value = 10)
    # Add more fields based on your model's requirements

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Upload File', required=True)
