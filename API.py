import matplotlib
matplotlib.use('Agg')
import initialization
import validation
import data_gathering
from flask import Flask, jsonify, request
from flask_cors import CORS
import charts
import threading
import logging
import app as sentiment 



# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the responses file
initialization.initialize_responses_file()

# API Endpoints
@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    try:
        return validation.fetch_data()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return jsonify({"status": "error", "message": "Failed to fetch data."}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    print(email, password)

    try:
        confimed = validation.login(email, password)
        if confimed:
            return jsonify({"status": "success", "message": "Login successful."}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid credentials."}), 401
    except Exception as e:
        logging.error(f"Error logging in: {e}")
        return jsonify({"status": "error", "message": "Failed to log in."}), 500

@app.route('/save-data', methods=['POST'])
def save_data():
    try:
        result = data_gathering.save_data()
        # Regenerate charts after saving data
        charts.plot_pie_charts()
        return result
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        return jsonify({"status": "error", "message": "Failed to save data."}), 500

@app.route('/update-data', methods=['PUT'])
def update_data():
    try:
        result = data_gathering.update_data()
        # Regenerate charts after updating data
        charts.plot_pie_charts()
        return result
    except Exception as e:
        logging.error(f"Error updating data: {e}")
        return jsonify({"status": "error", "message": "Failed to update data."}), 500

@app.route('/get-charts', methods=['GET'])
def get_charts():
    try:
        # Ensure the latest charts are served
        return jsonify({"charts": charts.cached_charts})
    except Exception as e:
        logging.error(f"Error fetching charts: {e}")
        return jsonify({"status": "error", "message": "Failed to fetch charts."}), 500

# @app.route('/analyze', methods=['GET'])
# def analyze():
#     # if not text:
#     #     return jsonify({'error': 'No text provided'}), 400
#     try:
#         result = sentiment.get_data()
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# Starting point for the app
def start_app():
    try:

        # Start file watcher in a daemon thread (unchanged)
        watcher_thread = threading.Thread(target=charts.start_watcher, daemon=True)
        watcher_thread.start()

        # Pre-generate charts (done in main thread)
        charts.plot_pie_charts()

        # Run the Flask app (this will run the app in the main thread)
        # app.run(debug=True, threaded=True)
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

    except Exception as e:
        logging.critical(f"Critical error starting the app: {e}")

if __name__ == "__main__":
    start_app()
