import pandas as pd
import csv

# adjust these for different output
predictions_path = "./GPT_Predictions.csv"
prediction_column_name = "GP4omini_Prediction"
output_path = "./GPT_Statistics.csv"

actual_column_name = "Type"
types = ["BOTH", "NONE", "ELECTION", "NONELECTION"]
output_headers = ["Type", "Precision", "Recall", "F1"]

def get_statistics_one_type(df, type):
    precision = get_precision(df, type)
    recall = get_recall(df, type)
    return precision, recall

def get_overall_statistics(df):
    results = {}
    total_weighted_f1 = 0
    total_size = 0
    for type in types:
        precision, recall = get_statistics_one_type(df, type)
        f1 = 2 * precision * recall / (precision + recall)
        results[type] = [precision, recall, f1]
        class_size = get_class_size(df, type)
        total_weighted_f1 += f1 * class_size
        total_size += class_size
    weighted_f1 = total_weighted_f1 / total_size
    results["overall_f1"] = weighted_f1
    return results

def get_class_size(df, type):
    return (df[actual_column_name] == type).sum()

def get_precision(df, type):
    true_pos = ((df[actual_column_name] == type) & (df[prediction_column_name] == type)).sum()
    false_pos = ((df[actual_column_name] != type) & (df[prediction_column_name] == type)).sum()
    return true_pos/(true_pos + false_pos)
    
def get_recall(df, type):
    true_pos = ((df[actual_column_name] == type) & (df[prediction_column_name] == type)).sum()
    false_neg = ((df[actual_column_name] == type) & (df[prediction_column_name] != type)).sum()
    return true_pos/(true_pos + false_neg)

def save_results(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(output_headers)
        for type in types:
            writer.writerow([type] + results[type])
        writer.writerow(["Overall", "", "", results["overall_f1"]])
        
if __name__ == "__main__":
    df = pd.read_csv(predictions_path)
    results = get_overall_statistics(df)
    save_results(results, output_path)