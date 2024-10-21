import pandas as pd
import kagglehub
import os
import shutil
import json
import sys

# Save the path to the dataset directory in a JSON file
def saveDataPath(dataset_dir):
    with open(json_file_path, 'w') as file:
        json.dump(dataset_dir, file, indent=4) #.dump() overwrites old json file and saves the new data

# Define the path to the JSON file
json_file_path = 'recidivismdataPath.json'
# Load the list from the JSON file if it exists, otherwise use the default list
if os.path.exists(json_file_path):
    with open(json_file_path, 'r') as file:
        dataset_dir = json.load(file)
else:
    dataset_dir = kagglehub.dataset_download("danofer/compass")
    saveDataPath(dataset_dir)

dataset_dir = "/path/to/dataset_directory"  # Define the directory where datasets are downloaded

def download_and_refresh_dataset():
    # Delete old dataset if it exists
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        print(f"Old dataset deleted from {dataset_dir}")

    # Download the new dataset
    path = kagglehub.dataset_download("danofer/compass")
    # Test print to see where the dataset is downloaded. Path
    print("New dataset downloaded to:", path)
    saveDataPath(path)

    return path

# Download fresh dataset and delete old one
new_path = download_and_refresh_dataset()

# List files in the directory
files = os.listdir(new_path)
print("Files in directory:", files)

csv_file_name = 'cox-violent-parsed_filt.csv'
# Find the CSV file specified!
csv_file = None
for file in files:
    if file == csv_file_name:
        csv_file = file
        break

if csv_file:
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(new_path, csv_file)
    print("CSV file path:", csv_file_path)

    # Read the CSV file
    data = pd.read_csv(csv_file_path)
    filtered_data = data[data['is_recid'] != -1]
    # print("Data loaded successfully")
    # print(filtered_data)
else:
    print("No CSV file found in the directory")
    sys.exit(0)

def get_data():
    return filtered_data

