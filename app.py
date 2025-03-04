from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Summarize the text
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

if __name__ == '__main__':
    app.run(debug=True)