import re

def process_file(path):
    lit = []
    file = open(path, "r")
    for line in file:
        text = line.strip()
        if is_final_annotation(line):
            continue
        if not text: 
            continue
        print(remove_intext_annotation(text))

def clear_annotation_markers(text:str) -> str:
    return re.sub(r'\{.*?\}', '', text)

def is_final_annotation(text:str) -> bool:
    return re.match(r"^\[[a-zA-Z]\]", text)

def remove_intext_annotation(text:str) -> str:
    temp_text = remove_curly_brackets(text)
    result = remove_square_brackets(temp_text)
    return result

def remove_curly_brackets(text:str) -> str:
    return re.sub(r'\{.*?\}', '', text)

def remove_square_brackets(text:str)-> str:
    return re.sub(r'\[.*?\]', '', text)

process_file("C:/Users/Dake Peng/Desktop/DataSquad/datasquad/all_docs/(AD) Copy of Reconciled Notation of KFXA 2016-10-23 08_58_04PM FOX 28 NEWS AT 9.txt")
