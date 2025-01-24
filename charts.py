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
import finalscript

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
choices = {
    "Q1": ["BS Computer Science", "BS Information Systems"],
    "Q2": ["1st year", "2nd year", "3rd year", "4th year"],
    "Q3": ["Regular student", "Irregular student", "Shiftee", "Transferee", "Returnee", "Working student", "Scholar", "Student assistant"] ,
    "Q4": ["Very satisfied", "Satisfied", "Dissatisfied", "Very dissatisfied"],
    "Q5": ["Academic courses", "Extracurricular activities", "Faculty interaction", "Peer interaction", "Campus facilities", "Support services"] ,
    "Q6": ["Academic workload", "Time management", "Financial issues", "Personal issues", "Lack of support services", "Health-related issues"],
    "Q7": ["Significantly improved", "Somewhat improved", "Remained the same", "Somewhat worsened", "Significantly worsened"],
    "Q8": ["Very supported", "Somewhat supported", "Supported", "Not supported"],
    "Q10": ['1','2','3','4','5'],
    "Q11": ['1','2','3','4','5'],
}

# print(*[item for values_list in choices.values() for item in values_list], sep='\n')
# for key, values_list in choices.items():
#     for item in values_list:
#         print(item)
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
                    textprops={'fontsize': 12},
                    colors=plt.cm.Paired.colors,
                    ax=ax,
                    labels=None,
                )
                # legend_items = [item for key, values_list in choices.items() if key == column for item in values_list]
                # # legend = pd.DataFrame(legend_items)
                # # legend = [item for legend_items.keys() in legend_items for legend_items.keys() in legend_items]
                
                # legend = [value for sublist in [value for key, value in choices.items() if key == column] for value in sublist]
                # print(legend)
                ax.legend(
                    sorted_responses.index,
                    # sorted_responses.index,
                    loc="upper right",
                    bbox_to_anchor=(1.25, 1),
                    fontsize=10,
                )
                plt.title(f"{questions.get(column, column)}", loc="left", fontsize=16, fontweight="bold")
                plt.ylabel("")
                plt.xlabel("")
            
                plt.title(f"{questions.get(column, column)}", loc="left", fontsize=16, fontweight="bold")
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
            elif column == "Q9":
                charts_data.append(finalscript.generate_wordcloud_positive())
            elif column == "Q12":
                charts_data.append(finalscript.generate_wordcloud_negative())
        # Update cached charts thread-safely
        with chart_lock:
            cached_charts.clear()  # Clear previous charts before updating
            cached_charts.extend(charts_data)  # Update the cache with new charts

        logging.info("Pie charts generated and cached successfully.")
        return charts_data

    except Exception as e:
        logging.error(f"Error generating pie charts: {e}")
        return []

# plot_pie_charts()
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
