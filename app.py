import os
import io
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, after_this_request, redirect, url_for
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis")

# Global variables to hold paths of temporary files
global_pie_chart_path = None
global_bar_chart_path = None
global_csv_path = None

@app.route("/", methods=["GET", "POST"])
def home():
    global global_pie_chart_path
    global global_bar_chart_path
    global global_csv_path

    sentiment = None

    if request.method == "POST":
        # Check if the text form was submitted
        if request.form.get('text'):
            text = request.form['text']
            try:
                result = nlp(text)
                sentiment = result[0]['label']
            except Exception as e:
                sentiment = f"Error in processing text: {str(e)}"

        # Check if a file was submitted
        elif 'csvfile' in request.files:
            file = request.files['csvfile']
            if file.filename.endswith('.csv'):
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file)

                    # Ensure there is a 'review' column
                    if 'review' in df.columns:
                        # Define a function to get the sentiment
                        def get_sentiment(text):
                            result = nlp(text)
                            return result[0]['label']

                        # Generate sentiment predictions for each review
                        df['sentiment'] = df['review'].apply(get_sentiment)

                        # Save the DataFrame to a temporary CSV file
                        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
                        os.close(temp_fd)  # Close the file descriptor
                        df.to_csv(temp_path, index=False)

                        # Set the global CSV path
                        global_csv_path = temp_path

                        # Create sentiment counts
                        sentiment_counts = df['sentiment'].value_counts()

                        # Create a pie chart from the sentiment data
                        plt.figure(figsize=(6, 6))
                        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336'])
                        plt.title('Sentiment Distribution')

                        # Save the pie chart to a temporary file
                        temp_fd, temp_pie_chart_path = tempfile.mkstemp(suffix='.png')
                        os.close(temp_fd)  # Close the file descriptor
                        plt.savefig(temp_pie_chart_path, format='png', bbox_inches='tight')
                        plt.close()

                        # Set the global pie chart path
                        global_pie_chart_path = temp_pie_chart_path

                        # Create a bar chart from the sentiment data
                        plt.figure(figsize=(6, 6))
                        sentiment_counts.plot(kind='bar', color=['#4CAF50', '#F44336'])
                        plt.title('Sentiment Count')
                        plt.xlabel('Sentiment')
                        plt.ylabel('Count')

                        # Save the bar chart to a temporary file
                        temp_fd, temp_bar_chart_path = tempfile.mkstemp(suffix='.png')
                        os.close(temp_fd)  # Close the file descriptor
                        plt.savefig(temp_bar_chart_path, format='png', bbox_inches='tight')
                        plt.close()

                        # Set the global bar chart path
                        global_bar_chart_path = temp_bar_chart_path

                        return redirect(url_for('show_chart'))

                    else:
                        sentiment = "CSV must contain a 'review' column."
                except Exception as e:
                    sentiment = f"Error processing CSV file: {str(e)}"
            else:
                sentiment = "Please upload a valid CSV file."

    return render_template("index.html", sentiment=sentiment)

@app.route("/show_chart")
def show_chart():
    global global_pie_chart_path
    global global_bar_chart_path
    global global_csv_path

    if global_pie_chart_path and global_bar_chart_path:
        return render_template("show_chart.html", global_pie_chart_path=global_pie_chart_path, global_bar_chart_path=global_bar_chart_path, global_csv_path=global_csv_path)

    return "No charts available", 404

@app.route("/chart_img/<chart_type>")
def show_chart_img(chart_type):
    global global_pie_chart_path
    global global_bar_chart_path

    chart_path = global_pie_chart_path if chart_type == 'pie' else global_bar_chart_path

    if chart_path:
        @after_this_request
        def cleanup(response):
            try:
                os.remove(chart_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")
            return response

        return send_file(chart_path, mimetype='image/png')

    return "No chart available", 404

@app.route("/download_csv")
def download_csv():
    global global_csv_path

    if global_csv_path:
        @after_this_request
        def cleanup(response):
            try:
                os.remove(global_csv_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")
            return response

        return send_file(global_csv_path, mimetype='text/csv', as_attachment=True, download_name='analyzed_reviews.csv')

    return "No CSV file available", 404

if __name__ == "__main__":
    app.run(debug=True)
