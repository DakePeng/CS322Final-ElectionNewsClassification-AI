import json
def save_json(var, output_path):
    with open(output_path, 'w') as file:
        json.dump(var, file, indent = 4)
        
def load_json(path):
    with open(path, 'r') as file:
        results = json.load(file)
        return results