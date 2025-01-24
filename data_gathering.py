from flask import Flask, request, jsonify
import csv
from datetime import datetime  # For timestamping backups
from flask_cors import CORS
import files

def save_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Append data to the responses file
        with open(files.RESPONSES_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data.values())

        # # Save a backup of the responses file
        # save_backup()

        return jsonify({"message": "Data saved successfully"}), 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    

def update_data():
    try:
        data = request.get_json()
        email = data.get("email")
        new_remarks = data.get("newRemarks", "Yes")
        field_to_update = "Remarks (Yes/No)"
        timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if not email or not field_to_update:
            return jsonify({"error": "Missing required fields"}), 400

        updated = False
        updated_rows = []

        # Read and update the masterlist
        with open(files.MASTERLIST_FILE, mode="r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

            for row in rows:
                if row.get("Email Address") == email:
                    row[field_to_update] = new_remarks
                    row["Timestamp"] = timestamp
                    updated = True
                updated_rows.append(row)

        if not updated:
            return jsonify({"error": "Email not found"}), 404

        # Write updated data back to the masterlist
        with open(files.MASTERLIST_FILE, mode="w", newline="") as file:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        return jsonify({"message": "Data updated successfully"}), 200
    except FileNotFoundError:
        return jsonify({"error": "Masterlist file not found"}), 404
    except Exception as e:
        print(f"Error updating data: {e}")
        return jsonify({"error": "Internal Server Error"}), 500