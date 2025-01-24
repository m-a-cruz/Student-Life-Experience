from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import langid
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

app = Flask(__name__)
CORS(app)

# Initialize NLP components
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
translator = Translator()

# Load lexicon data
try:
    lexicon_df = pd.read_csv('sentiment-Analyzer/data/lexdict.csv')
    lexicon = {row['word']: row['sentiment_score'] for _, row in lexicon_df.iterrows()}
except FileNotFoundError:
    print("Error: Lexicon file not found. Ensure 'sentiment-Analyzer/data/lexdict.csv' exists.")
    lexicon = {}

# Helper function to detect language
def detect_language(text):
    try:
        return langid.classify(text)[0]
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        return 'en'  # Default to English if detection fails

# Helper function to translate text to English
def translate_to_english(text):
    try:
        lang = detect_language(text)
        if lang in ['tl', 'bik']:  # Specific handling for Tagalog or Bikol
            return translator.translate(text, dest='en').text.lower()
        return text.lower()
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text.lower()  # Fallback to the original text if translation fails

# Preprocess text (cleaning, tokenization, stemming)
def process_text(text):
    # Clean and translate
    cleaned = re.sub(r'[^\w\s]', '', text).lower()  # Remove punctuation
    translated = translate_to_english(cleaned)

    # Tokenize and filter stopwords
    tokens = translated.split()
    tokens = [t for t in tokens if t not in stop_words]

    # Apply stemming
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

# Sentiment analysis logic
def analyze_sentiment(tokens):
    score = 0
    translation_stats = {
        'bikol_translated': 0,
        'tagalog_translated': 0,
        'total_translated': 0,
        'total_words': len(tokens)
    }

    for token in tokens:
        if token in lexicon:
            score += lexicon[token]
        else:
            try:
                # Attempt to translate unknown words
                lang = detect_language(token)
                if lang == 'bik':
                    translation_stats['bikol_translated'] += 1
                elif lang == 'tl':
                    translation_stats['tagalog_translated'] += 1

                translation_stats['total_translated'] += 1
                translated = translator.translate(token, dest='en').text.lower()
                score += lexicon.get(translated, 0)
            except Exception as e:
                print(f"Error processing token '{token}': {str(e)}")
                continue

    return score, translation_stats

# Flask route to analyze sentiment for CSV data
@app.route('/analyze', methods=['GET'])
def analyze():
    csv_file = request.args.get('file', 'csv_files/responses.csv')  # Default file name if not provided
    text_column = request.args.get('column', 'Q12')  # Default column name if not provided

    try:
        # Load the CSV file
        data = pd.read_csv(csv_file, encoding='ISO-8859-1')

        if text_column not in data.columns:
            return jsonify({'error': f"Column '{text_column}' not found in the CSV file."}), 400

       
        all_tokens = []

        
        for text in data[text_column]:
            tokens = process_text(str(text)) 
            all_tokens.extend(tokens)  

        overall_score, overall_stats = analyze_sentiment(all_tokens)

        overall_label = 'positive' if overall_score > 0 else 'negative' if overall_score < 0 else 'neutral'

        return jsonify({
            'overall_score': overall_score,
            'overall_label': overall_label,
            'overall_stats': overall_stats,
            'tokens': all_tokens  # Optionally include the tokens in the response
        })

    except FileNotFoundError:
        return jsonify({'error': f"CSV file '{csv_file}' not found."}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main execution block
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
