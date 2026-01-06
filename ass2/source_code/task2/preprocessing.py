
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
# Exclude negation words to preserve sentiment context
negations = {'no', 'not', 'nor', 'never', 'none', 'nothing'}
stop_words = stop_words.difference(negations)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references
    text = re.sub(r'@\w+', '', text)
    
    # Remove special characters and numbers (keeping only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization (split by space)
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return " ".join(tokens)

def preprocess_dataframe(df, text_column='text'):
    print("Cleaning text...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    return df
