import pandas as pd
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import langid  # Critical missing import

# Ensure you have downloaded the stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
translator = Translator()

# --- Enhanced Translation Function ---
def translate_to_english(text):
    """Translate full sentences with context preservation"""
    try:
        lang, _ = langid.classify(text)
        if lang in ['tl', 'bik']:  # Philippine languages
            translated = translator.translate(text, dest='en').text
            return translated.lower()
        return text.lower()
    except Exception as e:
        print(f"Translation error: {text} - {str(e)}")
        return text.lower()

# --- Language Detection Helper ---
def detect_language(text):
    try:
        lang, _ = langid.classify(text)
        return lang
    except:
        return 'en'

# --- Revised Processing Pipeline ---
def process_text(text):
    # Clean text
    cleaned = re.sub(r'[^\w\s]', '', text).lower()

    # Detect original language
    orig_lang = detect_language(cleaned)

    # Translate entire sentence
    translated = translate_to_english(cleaned)

    # Tokenize and process
    tokens = translated.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens, orig_lang

# --- Word Translation Tracking ---
def track_translated_words(tokens, lexicon):
    """Count words needing translation per language using the lexicon"""
    bikol_count = 0
    tagalog_count = 0
    total_needing_translation = 0

    for token in tokens:
        # Check if word exists in lexicon
        if token in lexicon:
            lang = lexicon[token]['language']
            if lang == 'bk':
                bikol_count += 1
            elif lang == 'tl':
                tagalog_count += 1
            total_needing_translation += 1
        else:
            # Word needs translation
            total_needing_translation += 1

    return bikol_count, tagalog_count, total_needing_translation

# --- Hybrid Sentiment Analysis ---
def sentiment_analysis(tokens, lexicon):
    score = 0
    for token in tokens:
        if token in lexicon:
            score += lexicon[token]['sentiment_score']
        else:
            try:
                translated = translate_to_english(token)
                if translated in lexicon:
                    score += lexicon.get(translated, {}).get('sentiment_score', 0)
            except Exception as e:
                print(f"Error processing token '{token}': {e}")
                continue
    return score

# --- Main Execution Flow ---
# Initialize tracking variables
total_bikol_translated = 0
total_tagalog_translated = 0
total_translated_words = 0
total_words_needing_translation = 0

# Load data
responses_df = pd.read_csv('csv_files/responses.csv', encoding='ISO-8859-1')
lexicon_df = pd.read_csv('csv_files/dictionary.csv', encoding='ISO-8859-1')
model_df = pd.read_csv('csv_files/model_parameters.csv', encoding='ISO-8859-1')

# Create enhanced lexicon dictionary
lexicon = {}
for _, row in lexicon_df.iterrows():
    lexicon[row['word']] = {
        'sentiment_score': row['sentiment_score'],
        'language': row['language']
    }

# Process narratives
narratives = responses_df[['Q9', 'Q12']].dropna()
narratives['combined'] = narratives['Q9'].astype(str) + ' ' + narratives['Q12'].astype(str)

results = []
for narrative in narratives['combined']:
    # Process text and get original language
    tokens, orig_lang = process_text(narrative)

    # Track translations using lexicon
    bikol, tagalog, total_needing = track_translated_words(tokens, lexicon)
    total_bikol_translated += bikol
    total_tagalog_translated += tagalog
    total_translated_words += len(tokens)
    total_words_needing_translation += total_needing

    # Calculate sentiment
    sentiment_score = sentiment_analysis(tokens, lexicon)
    sentiment_label = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'

    results.append({
        'original_text': narrative,
        'processed_text': ' '.join(tokens),
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label
    })

# Convert results to DataFrame
processed_df = pd.DataFrame(results)
processed_df.to_csv('narrative_sentiment.csv', index=False)

# --- Model Evaluation ---
score_to_label = {
    1: 'positive',
    0: 'neutral',
    -1: 'negative'
}

true_labels = [score_to_label[1] if row['coefficient'] > 0
               else score_to_label[-1] if row['coefficient'] < 0
               else score_to_label[0] for _, row in model_df.iterrows()][:len(results)]

predicted_labels = processed_df['sentiment_label'].values[:len(true_labels)]

# Use TF-IDF for feature representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_df['processed_text'])

# Train an SVM classifier with experimentation for kernel options
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
svm_classifier.fit(X, true_labels)

# Predict sentiment on the processed data
y_pred = svm_classifier.predict(X)

# Confusion Matrix and Accuracy
conf_matrix = confusion_matrix(true_labels, y_pred, labels=['positive', 'neutral', 'negative'])
accuracy = accuracy_score(true_labels, y_pred)
report = classification_report(true_labels, y_pred, labels=['positive', 'neutral', 'negative'], target_names=['positive', 'neutral', 'negative'], zero_division=0)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# --- Visualization ---
def display_sentiment_distribution(predictions):
    """
    Display a bar graph of the sentiment distribution.

    Args:
        predictions (list): A list of predicted sentiment labels.
    """
    sentiment_counts = pd.Series(predictions).value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution')
    plt.show()

def generate_sentiment_wordcloud(processed_df):
    """
    Generate word clouds for positive and negative sentiments.

    Args:
        processed_df (DataFrame): The DataFrame containing processed text and sentiment scores.
    """
    positive_words = ' '.join(processed_df[processed_df['sentiment_score'] > 0]['processed_text'])
    negative_words = ' '.join(processed_df[processed_df['sentiment_score'] < 0]['processed_text'])

    plt.figure(figsize=(12, 6))

    # Positive sentiment word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(WordCloud(width=400, height=400, background_color='white').generate(positive_words if positive_words else "No Positive Words"))
    plt.title('Positive Words')
    plt.axis('off')

    # Negative sentiment word cloud
    plt.subplot(1, 2, 2)
    plt.imshow(WordCloud(width=400, height=400, background_color='white').generate(negative_words if negative_words else "No Negative Words"))
    plt.title('Negative Words')
    plt.axis('off')

    plt.show()

display_sentiment_distribution(y_pred)
generate_sentiment_wordcloud(processed_df)

# Print translation statistics
print(f"Total Bikol words translated: {total_bikol_translated}")
print(f"Total Tagalog words translated: {total_tagalog_translated}")
print(f"Total words translated: {total_translated_words}")
print(f"Translation Accuracy: {total_translated_words / total_words_needing_translation * 100:.2f}%")

print("\nSentiment analysis completed successfully.")