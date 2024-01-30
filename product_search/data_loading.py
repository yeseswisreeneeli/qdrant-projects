import json

# Function to load JSON data from the file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
    
    
