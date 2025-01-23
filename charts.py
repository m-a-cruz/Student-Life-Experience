import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import logging
import time
import files

# Global variables
questions = {
    "Q1": "Program",
    "Q2": "Level",
    "Q3": "Status",
    "Q4": "Experience Satisfaction",
    "Q5": "Personal Growth",
    "Q6": "Challenges",
    "Q7": "Improvements",
    "Q8": "Academic and Personal Goals",
    "Q10": "Teaching and Learning Satisfaction",
    "Q11": "Other Facilities Satisfaction",
}

# Cached charts and lock for thread-safety
cached_charts = []
chart_lock = threading.Lock()
RESPONSES_FILE = files.RESPONSES_FILE  # Update with the correct path

def plot_pie_charts():
    """
    Generates pie charts for survey responses and caches them.
    """
    global cached_charts
    try:
        data = pd.read_csv(RESPONSES_FILE, encoding='ISO-8859-1')
        charts_data = []

        for column in data.columns:
            if column not in ["Q9", "Q12"]:  # Exclude specified columns
                separated_list = []

                # Handle multi-response columns
                if column in ["Q3", "Q5", "Q6"]:
                    for response in data[column].dropna():
                        separated_list.extend(response.split(", "))
                    response_counts = pd.Series(separated_list).value_counts()
                else:
                    response_counts = data[column].value_counts()

                sorted_responses = response_counts.sort_index()

                # Plot pie chart
                fig, ax = plt.subplots(figsize=(7, 6))
                plt.subplots_adjust(left=-0.1, right=0.9, top=0.9, bottom=0, hspace=0.4)

                sorted_responses.plot.pie(
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 10},
                    colors=plt.cm.Paired.colors,
                    ax=ax,
                    labels=None,
                )

                ax.legend(
                    sorted_responses.index,
                    loc="upper left",
                    bbox_to_anchor=(0.85, 1),
                    fontsize=10,
                )

                plt.title(f"{questions.get(column, column)}", loc="left", fontsize=12, fontweight="bold")
                plt.ylabel("")
                plt.xlabel("")

                # Convert chart to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                buffer.close()
                plt.close(fig)

                charts_data.append({"column": column, "image": img_base64})

        # Update cached charts thread-safely
        with chart_lock:
            cached_charts.clear()  # Clear previous charts before updating
            cached_charts.extend(charts_data)  # Update the cache with new charts

        logging.info("Pie charts generated and cached successfully.")
        return charts_data

    except Exception as e:
        logging.error(f"Error generating pie charts: {e}")
        return []

class FileWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(RESPONSES_FILE):
            logging.info("CSV file updated, regenerating charts...")
            plot_pie_charts()  # Regenerate the charts
            logging.info("Charts regenerated.")


def start_watcher():
    """
    Starts a file watcher in a separate thread.
    """
    event_handler = FileWatcher()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()
    logging.info("File watcher started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
