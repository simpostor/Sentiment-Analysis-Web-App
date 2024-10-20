from flask import Flask, request, render_template, send_file, after_this_request
from transformers import pipeline
import pandas as pd
import os
import tempfile

app = Flask(__name__)

# Load the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None

    if request.method == "POST":
        # Check if the text form was submitted
        if request.form.get('text'):
            text = request.form['text']
            print(f"Text input received: {text}")  # Debugging print
            try:
                result = nlp(text)
                sentiment = result[0]['label']
                print(f"Sentiment result: {sentiment}")  # Debugging print
            except Exception as e:
                sentiment = f"Error in processing text: {str(e)}"
                print(f"Error: {str(e)}")  # Error logging

        # Check if a file was submitted
        elif 'csvfile' in request.files:
            file = request.files['csvfile']
            print(f"File received: {file.filename}")  # Debugging print
            if file.filename.endswith('.csv'):
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file)

                    # Ensure there is a 'review' column
                    if 'review' in df.columns:
                        print(f"'review' column found in CSV")  # Debugging print

                        # Define a function to get the sentiment
                        def get_sentiment(text):
                            result = nlp(text)
                            return result[0]['label']

                        # Generate sentiment predictions for each review
                        df['sentiment'] = df['review'].apply(get_sentiment)

                        # Create a temporary file to save the output CSV
                        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
                        os.close(temp_fd)  # Close the file descriptor

                        df.to_csv(temp_path, index=False)
                        print(f"CSV saved to temp path: {temp_path}")  # Debugging print

                        @after_this_request
                        def cleanup(response):
                            try:
                                os.remove(temp_path)
                                print(f"Temporary file {temp_path} removed")  # Debugging print
                            except Exception as e:
                                print(f"Error removing temp file: {e}")
                            return response

                        return send_file(temp_path, as_attachment=True, download_name='output_with_sentiments.csv')

                    else:
                        sentiment = "CSV must contain a 'review' column."
                        print("Error: CSV does not contain 'review' column.")  # Error logging
                except Exception as e:
                    sentiment = f"Error processing CSV file: {str(e)}"
                    print(f"Error processing CSV file: {str(e)}")  # Error logging
            else:
                sentiment = "Please upload a valid CSV file."
                print("Invalid file type. Only .csv files are allowed.")  # Error logging

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
