import os
import csv
from datetime import datetime

def log_to_csv(query, answer):
    filename = "qa_log.csv"
    
    # Check if the file doesn't exist, to write headers
    write_header = not os.path.exists(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Timestamp", "Question", "Answer"])
        writer.writerow([datetime.now(), query, answer])