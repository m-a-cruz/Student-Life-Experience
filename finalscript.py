import pandas as pd
import re
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
import joblib

# Ensure you have downloaded the stopwords
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
translator = Translator()

# --- Configuration ---
DATA_PATHS = {
    'responses': 'responses1.csv',
    'lexicon': 'dictionary.csv',
    'model': 'model_parameters.csv',
    'output': 'narrative_sentiment.csv'
}

# --- 1. Load Data ---
def load_data():
    """Load responses, lexicon, and model parameters from CSV files."""
    try:
        responses_df = pd.read_csv(DATA_PATHS['responses'], encoding='latin1')  # Specify encoding
        lexicon_df = pd.read_csv(DATA_PATHS['lexicon'], encoding='latin1')  # Specify encoding
        model_df = pd.read_csv(DATA_PATHS['model'], encoding='latin1')  # Specify encoding
        
        # Convert lexicon to dictionary {word: sentiment_score}
        lexicon = {row['word']: row['sentiment_score'] for _, row in lexicon_df.iterrows()}
        return responses_df, lexicon, model_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# --- 2. Text Processing Functions ---
def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    tokens = [stemmer.stem(token) for token in tokens]  # Perform stemming
    return tokens

def cleanse_text(tokens):
    """Cleanse the tokens."""
    cleaned_tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    cleaned_tokens = [token for token in cleaned_tokens if len(token) > 2]  # Remove short words
    return cleaned_tokens

def safe_translate(token):
    """Safely translate a token with error handling."""
    try:
        lang = detect(token)
        if lang in ['tl', 'bk']:  # Tagalog or Bikol
            return translator.translate(token, dest='en').text.lower()
        return token
    except Exception as e:
        logging.error(f"Error translating token '{token}': {e}")
        return token

def translate_to_english(tokens):
    """Translate tokens to English and count translated words."""
    translated_tokens = []
    bikol_count = tagalog_count = total_translated = total_words_needing_translation = 0

    for token in tokens:
        translated_token = safe_translate(token)
        translated_tokens.append(translated_token)
        if translated_token != token:
            total_translated += 1
            if detect(token) == 'tl':
                tagalog_count += 1
            else:
                bikol_count += 1

    total_words_needing_translation = len(tokens)
    return translated_tokens, bikol_count, tagalog_count, total_translated, total_words_needing_translation

def map_sentiment_to_label(sentiment_score):
    """Map sentiment score to label."""
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score == 0:
        return 'neutral'
    else:
        return 'negative'

def sentiment_analysis(tokens, lexicon):
    """Perform sentiment analysis on the tokens."""
    sentiment_score = sum(lexicon.get(token, 0) for token in tokens)
    return sentiment_score

# ```python
# --- 3. Process Data and Perform Sentiment Analysis ---
def process_narratives(responses_df, lexicon):
    """Process narratives and perform sentiment analysis."""
    narratives = responses_df[['Q9', 'Q12']].dropna()
    narratives['combined'] = narratives['Q9'].astype(str) + ' ' + narratives['Q12'].astype(str)

    results = []
    total_bikol_translated = 0
    total_tagalog_translated = 0
    total_translated_words = 0
    total_words_needing_translation = 0

    for narrative in narratives['combined']:
        tokens = preprocess_text(narrative)
        cleaned_tokens = cleanse_text(tokens)
        translated_tokens, bikol_count, tagalog_count, total_count, total_needing_translation = translate_to_english(cleaned_tokens)
        sentiment_score = sentiment_analysis(translated_tokens, lexicon)
        sentiment_label = map_sentiment_to_label(sentiment_score)

        results.append({
            'original_text': narrative,
            'processed_text': ' '.join(translated_tokens),
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        })

        total_bikol_translated += bikol_count
        total_tagalog_translated += tagalog_count
        total_translated_words += total_count
        total_words_needing_translation += total_needing_translation

    return results, total_bikol_translated, total_tagalog_translated, total_translated_words, total_words_needing_translation

# --- 4. Model Evaluation ---
def evaluate_model(processed_df, model_df):
    """Evaluate the model's performance using the processed data."""
    score_to_label = {
        1: 'positive',
        0: 'neutral',
        -1: 'negative'
    }

    true_labels = [score_to_label[1] if row['coefficient'] > 0
                   else score_to_label[-1] if row['coefficient'] < 0
                   else score_to_label[0] for _, row in model_df.iterrows()][:len(processed_df)]

    predicted_labels = processed_df['sentiment_label'].values[:len(true_labels)]

    # Use TF-IDF for feature representation
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_df['processed_text'])

    # Train an SVM classifier
    svm_classifier = SVC(kernel='linear')
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

    # Save the model
    joblib.dump(svm_classifier, 'sentiment_model.pkl')
    logging.info("Model saved as 'sentiment_model.pkl'.")

# --- 5. Visualization ---
def display_sentiment_distribution(predictions):
    """Display a bar graph of the sentiment distribution."""
    sentiment_counts = pd.Series(predictions).value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution')
    plt.show()

def generate_sentiment_wordcloud(processed_df):
    """Generate word clouds for positive and negative sentiments."""
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

# --- 6. Main Execution Flow ---
def main():
    """Main function to execute the sentiment analysis pipeline."""
    responses_df, lexicon, model_df = load_data()
    results, total_bikol_translated, total_tagalog_translated, total_translated_words, total_words_needing_translation = process_narratives(responses_df, lexicon)

    # Convert results to DataFrame and save
    processed_df = pd.DataFrame(results)
    processed_df.to_csv(DATA_PATHS['output'], index=False)

    # Evaluate the model
    evaluate_model(processed_df, model_df)

    # Visualization
    display_sentiment_distribution(processed_df['sentiment_label'])
    generate_sentiment_wordcloud(processed_df)

    # Print translation statistics
    print(f"Total Bikol words translated: {total_bikol_translated}")
    print(f"Total Tagalog words translated: {total_tagalog_translated}")
    print(f"Total words translated: {total_translated_words}")
    print(f"Translation Accuracy: {total_translated_words / total_words_needing_translation * 100:.2f}%")

    print("\nSentiment analysis completed successfully.")

if __name__ == "__main__":
    main()