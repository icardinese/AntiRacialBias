# AntiRacialBias

Rapid racial fairness identifier and corrector by taking a dataset of the Recidivism and Violent Recidivism Dataset COMPAS from Pro Publica and train it in the XGBOOST model to see the results. Disparities in result within different racial groups will be mitigated by examining FPR (False Positive Rates) and FNR (False Negative Rates) across different racial groups. Then equalized odds will be applied to equalize the false positive rates across racial groups. The results are displayed in a nice easy-to-use website where COMPAS like documents can be uploaded and factors can be fed into the model for prediction and a pdf report that can be taken to courst for mroe equitable decisions. This is necessary to reduce the racial divide and to create a more accurate algorithm based on a greater wealth of data and a multi-model solution.

## Setup Guide
A virtual environment is necessary so the program can run without any conflicts and with the correct dependencies. This makes the software replicable on any device.

**Setup the Virtual Environment**

A pre-requisite is to have python 3.12.4 specifically installed on your machine. Any other version I can't guarentee it will work smoothly

  1. Download the software and open powershell. (If you opened powershell through VS code's or any IDE's terminal then move on to step 2). Change the path of the powershell to the location of the folder.
```
cd (path)
```
For example my path is:
```
cd C:\Users\Owner\OneDrive\CODE\AntiRacialBias
```
  2. In the powershell type the following command:
```
py -m venv venv
```
  3. Activate the virtual environment (Only on windows):
```
.\venv\Scripts\Activate.ps1
```
  4. Install the dependencies and their corresponding version:
```
pip install -r requirements.txt
```
  5. Now download the AI models by referring to this google drive link:\
https://drive.google.com/drive/folders/1CyJdZ_KwPsd4oYzD6neZkl38x6smDncj?usp=sharing
  6. Put all these AI models right under the project directory and not in any folders. (This will mess up the paths)\
Example File Structure:\
```
your_project/ 
├── ai model 1
├── ai model 2
├── ai model 3
.\
.\
.\
├── data
├── evaluations
├── figuresAndVisualsforResearch
```
  8. Go into frontend and go into app.py\
```
your_project/ 
├───frontend
| └───__pycache__
| ├───static
| │   ├───images
| │   └───reports
| ├───templates
| ├── app.py  <<------ Go here 
| ├── forms.py 
| ├── model_pipeline.py
```
  9. Run app.py and in the terminal click on the link (cntrl + leftclick):
```
http://127.0.0.1:5000
```
You have finished setting up the pre-requisites for the program and can run it freely!!

