import pandas as pd
import csv

# adjust these for different output
predictions_path = "./results/GPT_Predictions.csv"
prediction_column_name = "GPT4omini_Prediction"
output_path = "./results/GPT_Statistics.csv"

actual_column_name = "Type"
types = ["NONE", "ELECTION", "NONELECTION"]
output_headers = ["Type", "Precision", "Recall", "F1"]

def get_statistics_one_type(df, type):
    precision = get_precision(df, type)
    recall = get_recall(df, type)
    return precision, recall

def get_actual_predicted_count(df, actual_type, predicted_type):
    return ((df[actual_column_name] == actual_type) & (df[prediction_column_name] == predicted_type)).sum()

def get_confusion_matrix(df):
    confusion_matrix = []
    for i in range(len(types)): 
        row = []
        type = types[i]
        for j in range(len(types)): 
            type2 = types[j]
            row.append(get_actual_predicted_count(df, type2, type))
        confusion_matrix.append(row)
    return confusion_matrix
    
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
    print(f"{type} {(df[actual_column_name] != type).sum()} {(df[prediction_column_name] == type).sum()}")
    true_pos = ((df[actual_column_name] == type) & (df[prediction_column_name] == type)).sum()
    false_pos = ((df[actual_column_name] != type) & (df[prediction_column_name] == type)).sum()
    return true_pos/(true_pos + false_pos)
    
def get_recall(df, type):
    true_pos = ((df[actual_column_name] == type) & (df[prediction_column_name] == type)).sum()
    false_neg = ((df[actual_column_name] == type) & (df[prediction_column_name] != type)).sum()
    return true_pos/(true_pos + false_neg)

def save_results(confusion_matrix, stats, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Confusion Matrix:"])
        writer.writerow(["Predicted/Actual"] + types)
        for i in range(len(confusion_matrix)):
            writer.writerow([types[i]] + confusion_matrix[i])
        writer.writerow([])
        writer.writerow(["Statistics:"])
        writer.writerow(output_headers)
        for type in types:
            writer.writerow([type] + stats[type])
        writer.writerow(["Overall", "", "", stats["overall_f1"]])
        
if __name__ == "__main__":
    df = pd.read_csv(predictions_path)
    confusion_matrix = get_confusion_matrix(df)
    stats = get_overall_statistics(df)
    save_results(confusion_matrix, stats, output_path)