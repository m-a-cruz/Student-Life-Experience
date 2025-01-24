from flask import Flask, request, jsonify
import csv
from flask_cors import CORS
import files
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


# Initialize the responses CSV with headers
def initialize_responses_file():
    """
    Initializes the responses CSV file with predefined headers.
    Writes headers only if the file doesn't exist.
    """
    headers = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12"]
    
    if not os.path.exists(files.RESPONSES_FILE):  # Check if the file exists
        try:
            # Create the file and write headers
            with open(files.RESPONSES_FILE, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                print(f"Responses file initialized with headers: {headers}")
        except Exception as e:
            print(f"Error initializing responses file at {files.RESPONSES_FILE}: {e}")
    else:
        print(f"Responses file already exists at {files.RESPONSES_FILE}, skipping initialization.")


