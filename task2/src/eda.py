import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def run_eda(df=None, input_file='data/sentiment140_raw.csv', output_dir='paper/figures'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if df is None:
        print("Loading data for EDA...")
        try:
            df = pd.read_csv(input_file)
            # Map sentiment 0->Negative, 4->Positive
            df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 4: 'Positive', 2: 'Neutral'})
        except FileNotFoundError:
            print("Data file not found. Skipping EDA.")
            return
    else:
        print("Using provided DataFrame for EDA...")
        if 'sentiment_label' not in df.columns and 'sentiment' in df.columns:
             df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 4: 'Positive', 2: 'Neutral'})

    print("Generating Sentiment Distribution Plot...")
    plt.figure(figsize=(8,6))
    sns.countplot(x='sentiment_label', data=df)
    plt.title('Sentiment Distribution')
    plt.savefig(f'{output_dir}/sentiment_distribution.png')
    plt.close()
    
    print("Generating Word Clouds...")
    # Sample to avoid memory issues if dataset is huge. 
    # If df is already passed (e.g. 200k sample), we might not need to sample again, but safe to do so for plotting.
    if len(df) > 100000:
        df_sample = df.sample(n=100000, random_state=42)
    else:
        df_sample = df
    
    # Determine text column to use
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    print(f"Using column '{text_col}' for Word Clouds.")
    
    # If using cleaned_text, we assume stopwords are already handled, so we disable WordCloud's internal stopword removal
    # to preserve terms like "not" if they were kept intentionally.
    stopwords_wc = set() if text_col == 'cleaned_text' else None

    # Positive
    pos_text = " ".join(df_sample[df_sample['sentiment'] == 4][text_col].astype(str))
    if pos_text:
        wc_pos = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_wc).generate(pos_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Sentiment Word Cloud')
        plt.savefig(f'{output_dir}/wordcloud_positive.png')
        plt.close()
        
    # Negative
    neg_text = " ".join(df_sample[df_sample['sentiment'] == 0][text_col].astype(str))
    if neg_text:
        print("Filtering negative text using VADER lexicon (Whitelist approach)...")
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            
        sia = SentimentIntensityAnalyzer()
        # Create a set of words that have a negative valence score
        negative_whitelist = {word for word, score in sia.lexicon.items() if score < 0}
        
        # Filter the text to keep ONLY words in the negative whitelist
        tokens = neg_text.split()
        filtered_tokens = [t for t in tokens if t in negative_whitelist]
        neg_text_filtered = " ".join(filtered_tokens)
        
        if not neg_text_filtered:
            print("Warning: Whitelist filtering resulted in empty text. Falling back to original.")
            neg_text_filtered = neg_text

        wc_neg = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(neg_text_filtered)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Sentiment Word Cloud (Pure Negative Terms)')
        plt.savefig(f'{output_dir}/wordcloud_negative.png')
        plt.close()
        
    print("EDA Complete. Figures saved to", output_dir)

if __name__ == "__main__":
    run_eda()
