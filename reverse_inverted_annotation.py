import os
from get_training_data import clean_concatenate_text

directory = './annotated_docs/'

def is_nonelection_markers_inverted(concatenated_text):
    find_beginning = concatenated_text.find("\\")
    find_ending= concatenated_text.find("//")
    if find_beginning == -1 and find_ending == -1:
        return False
    if find_ending == -1:
        return False
    if find_beginning < find_ending: 
        return False
    return True

def swap_nonelection_markers(path):
    temp_lines = []
    with open(path, 'r') as file:
        for line in file:
            inverted_line = line.replace('\\\\', 'TEMP_SWAP').replace('//', '\\\\').replace('TEMP_SWAP', '//')
            temp_lines.append(inverted_line)
    with open(path, 'w') as file:
        file.writelines(temp_lines)
        
if __name__ == "__main__":
    files = [(directory + f) for f in os.listdir(directory)]
    count = 0
    for path in files:
        file = open(path, "r", encoding='utf-8')
        concatenated_text = clean_concatenate_text(file)
        if is_nonelection_markers_inverted(concatenated_text):
            print(path)
            swap_nonelection_markers(path)
