import re
import os 
import csv
from tqdm import tqdm

EOL_MARKER = "ζ"
MIN_EXCERPT_LENGTH = 7
annotated_docs_folder = "./annotated_docs/"
training_csv = 'training_data.csv'
header = ["Source", "Date", "Text", "Type"]

class Excerpt:
    text = ""
    type = ""
    date = ""
    def __init__(self, text, type, date):
        self.text = text
        self.type = type
        self.date = date
    def __str__(self):
        return f"DATE: {self.date}; TYPE: {self.type}\nTEXT: {self.text}"

class TextBreak:
    index_start = 0
    index_end = 0
    type = None
    # Line Ends, Non-election story Starts, Non-election story Ends
    # Election story Starts, Election story Ends, Neither Ends, Both Ends
    possible_types = ["LE", "NS", "NE", "ES", "EE", "NN", "BT"]
    def __init__(self, start:int, end:int, type:str):
        if type not in self.possible_types:
            raise TypeError("Invalid Type")
        self.start = start 
        self.end = end 
        self.type = type
    def __str__(self):
        return f"({self.start}, {self.type})"

class Text_Piece:
    start = 0
    end = 0
    isElection = False
    isNonElection = False
    def __init__(self, start:int, end:int, isElection: bool, isNonElection: bool):
        self.start = start
        self.end = end
        self.isElection = isElection
        self.isNonElection = isNonElection
    def __str__(self):
        return f"(Election: {self.isElection} NonElection: {self.isNonElection})"

def get_excerpts_from_file(path):
    file = open(path, "r", encoding='utf-8')
    date = get_date_from_file(file)
    concatenated_text = clean_concatenate_text(file)
    excerpts = get_excerpts(concatenated_text, date)
    return excerpts

def get_excerpts(concatenated_text, date):
    breaks = get_breaks(concatenated_text)
    text_pieces = get_text_pieces(breaks)
    text_pieces = remove_empty_text_pieces(text_pieces, MIN_EXCERPT_LENGTH)
    text_pieces = merge_text_pieces(text_pieces, concatenated_text)
    return make_excerpts_list(text_pieces, concatenated_text, date)
    
def make_excerpts_list(text_pieces, concatenated_text, date):
    excerpts = []
    for text_piece in text_pieces:
        text = concatenated_text[text_piece.start: text_piece.end]
        text = clear_annotation_markers(text)
        if len(text) < MIN_EXCERPT_LENGTH:
            continue
        type = ""
        if text_piece.isElection and text_piece.isNonElection:
            type = "BOTH"
        elif text_piece.isElection:
            type = "ELECTION"
        elif text_piece.isNonElection:
            type = "NONELECTION"
        else:
            type = "NONE"   
        excerpts.append(Excerpt(text, type, date))  
    return excerpts     
            
def print_text_pieces(text_pieces, concatenated_text):
    for text in text_pieces:
        print(text)
        print(concatenated_text[text.start:text.end])
            
def remove_empty_text_pieces(text_pieces, thereshold):
    i = 0
    while i < (len(text_pieces) - 1) and text_pieces[i].end < thereshold: 
        text_pieces.pop(i)
    while i < (len(text_pieces) - 1):
        if text_pieces[i + 1].end - text_pieces[i].end < thereshold:
            text_pieces.pop(i + 1)
        else:
            i += 1
    return text_pieces

def merge_text_pieces(text_pieces, concatenated_text):    
    i = 0
    while i < (len(text_pieces) - 1):
        next_piece = text_pieces[i + 1]
        if text_pieces[i].isElection or text_pieces[i].isNonElection:
            j = text_pieces[i].end
            if text_pieces[i].isElection == next_piece.isElection and text_pieces[i].isNonElection == next_piece.isNonElection and concatenated_text[j] == EOL_MARKER:
                text_pieces[i].end = next_piece.end
                text_pieces.pop(i + 1)
                continue
        i += 1
    return text_pieces

def get_text_pieces(breaks):
    current_start = 0
    isElection = False
    isNonElection = False
    text_pieces = []
    for i in range(0, len(breaks)):
        br = breaks[i]
        text_pieces.append(Text_Piece(current_start, br.start, isElection, isNonElection))
        if br.type == "LE":
            # check if the annotator forgot to end the story.
            if find_next(breaks, i, "EE") == 0: isElection = False
            if find_next(breaks, i, "NE") == 0: isNonElection = False
            if find_next(breaks, i, "EE") > find_next(breaks, i, "ES"): isElection = False
            if find_next(breaks, i, "NE") > find_next(breaks, i, "NS"): isNonElection = False
        if br.type == "EE":
            isElection = False
        if br.type == "NE":
            isNonElection = False
        if br.type == "NS":
            isNonElection = True
        if br.type == "ES":
            isElection = True 
        current_start = br.end
    return text_pieces

def get_breaks(concatenated_text):
    breaks = []
    breaks += find_breaks(concatenated_text, EOL_MARKER, "LE")
    breaks += find_breaks(concatenated_text, r"//", "NE")
    breaks += find_breaks(concatenated_text, r"\\\\", "NS")
    breaks += find_breaks(concatenated_text, r"\~", "ES")
    breaks += find_breaks(concatenated_text, r"\|", "EE")
    breaks = sorted(breaks, key=lambda br: br.start)
    return breaks

def find_breaks(text, pattern, type):
    breaks = []
    for match in re.finditer(pattern, text):
        breaks.append(TextBreak(match.start(), match.end(), type))
    return breaks

def find_next(breaks, i, type):
    for j in range(i + 1, len(breaks)):
        if breaks[j].type == type:
            return j
    return 0 

def get_date_from_file(file):
    date = "N/A"
    for i, line in enumerate(file):
        if i == 1:  # Second line (0-based index)
            second_line = line.strip()
            date = " ".join(second_line.split()[0:3])
            break
    return date

def clean_concatenate_text(file):
    def is_final_annotation(text:str) -> bool:
        return re.match(r"^\[[a-zA-Z]\]", text)
    concatenated_text = ""
    for line in file:
        text = line.strip()
        # remove empty, starting, and final rows
        if not text or not text.startswith("[") or is_final_annotation(text):
            continue        
        text = remove_intext_annotation(text)
        concatenated_text += text + EOL_MARKER
    return concatenated_text

def clear_annotation_markers(text:str) -> str:
    return re.sub(r'//|\\\\|~|\||\ζ', '', text)

def remove_intext_annotation(text:str) -> str:
    def remove_curly_brackets(text:str) -> str:
        return re.sub(r'\{.*?\}', '', text)
    def remove_square_brackets(text:str)-> str:
        return re.sub(r'\[.*?\]', '', text)
    temp_text = remove_curly_brackets(text)
    result = remove_square_brackets(temp_text)
    return result

if __name__ == "__main__":
    annotated_files = os.listdir(annotated_docs_folder)
    with open(training_csv, mode='w', newline='', encoding='utf-8') as csv_output:
        writer = csv.writer(csv_output)
        writer.writerow(header)
        for path in tqdm(annotated_files):
            file_path = os.path.join(annotated_docs_folder, path)
            excerpts = get_excerpts_from_file(file_path)
            for excerpt in excerpts:
                data_row = [path, excerpt.date, excerpt.text, excerpt.type]
                writer.writerow(data_row)