from flask import Flask, request, jsonify
import csv
from flask_cors import CORS
import files
import bcrypt

def fetch_data():
    try:
        with open(files.MASTERLIST_FILE, mode="r") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]  # Convert rows to a list of dictionaries
        return jsonify(data), 200
    except FileNotFoundError:
        return jsonify({"error": "Masterlist file not found"}), 404
    except Exception as e:
        print(f"Error fetching data: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    
users = [
    {
        "email": "anna.romulo@chmsc.edu.ph",
        "password": bcrypt.hashpw("!Juan23456Seven".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')  # Hashed password
        # "password": "!Juan23456Seven",
    }
]

def login(email, password):
    for user in users:
        if user["email"] == email and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        # if user["email"] == email and user["password"] == password:    
            return True
    return False