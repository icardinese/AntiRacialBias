import os
file_path = 'C:\\Users\\Owner\\.cache\\kagglehub\\datasets\\uciml\\adult-census-income\\versions\\3'

# Check if the file exists
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")

# Check file permissions
print("Readable:", os.access(file_path, os.R_OK))
print("Writable:", os.access(file_path, os.W_OK))
print("Executable:", os.access(file_path, os.X_OK))