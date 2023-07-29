import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from io import StringIO

import pandas as pd
pd.set_option('display.max_colwidth', None)

from textblob import TextBlob
from wordcloud import WordCloud
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

from bs4 import BeautifulSoup
import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import random
import warnings
warnings.filterwarnings("ignore")



# text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text, stop_words):
    if text and isinstance(text, str):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+|[^a-zA-Z]', ' ', text.lower())
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if len(word) > 1 and word not in stop_words])
        text = ' '.join(list(dict.fromkeys(text.split())))
    else:
        text = ''
    return text

# Load the trained model
with open('./models/lr.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
with open('./models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def show_sentiment_detection():
    st.title("Sentiment Detection")
    new_sentence = st.text_area("Enter the text for sentiment analysis", "")
    
    analyze_clicked = st.button("Analyze")
       
    if analyze_clicked:
        if new_sentence:
            # Preprocess the user input
            cleaned_sentence = clean_text(new_sentence, stop_words)
            
            input_features = vectorizer.transform([cleaned_sentence])
            
            predicted_sentiment = model.predict(input_features)[0]

            # Get the probability scores for each sentiment category
            probability_scores = model.predict_proba(input_features)[0]
            probability_scores_dict = {model.classes_[i]: probability_scores[i] for i in range(len(model.classes_))}
            
            # Sort the probability scores from highest to lowest
            sorted_probabilities = sorted(probability_scores_dict.items(), key=lambda x: x[1], reverse=True)

            # Print the user input and predicted sentiment
            st.info(f"Your sentence : {new_sentence}")
            st.info(f"Cleaned Text : {cleaned_sentence}")
            st.info(f"Predicted sentiment : {sorted_probabilities[0][0]}")

            # Prepare data for bar chart
            sentiment_labels = [sentiment for sentiment, _ in sorted_probabilities]
            probabilities = [probability * 100 for _, probability in sorted_probabilities]

            # Display bar chart
            st.subheader("Sentiment Probabilities")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(sentiment_labels, probabilities)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Probability (%)")
            ax.set_title("Sentiment Probabilities")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            for i, v in enumerate(probabilities):
                ax.text(i, v, f"{v:.2f}%", color='black', ha='center')
            plt.xticks()
            st.pyplot(fig)
            
             # Save as image button
            save_as_image(cleaned_sentence, sorted_probabilities, fig)

            
        else:
            st.error("Please enter a sentence for sentiment analysis.")
            
def save_as_image(new_sentence, sorted_sentiments, fig):
    # Create a new PIL image with the text
    image = Image.new("RGB", (1200, 1200), "white")
    st_image = ImageDraw.Draw(image)
    font = ImageFont.truetype("./fonts/Arial.ttf", 18)
    # Split the new_sentence into a list of words
    words = new_sentence.split()

    # Group the words into lines of 14 words each
    lines = [words[i:i+14] for i in range(0, len(words), 14)]

    # Create a string with new lines for every 14 words
    sentence_lines = '\n'.join([' '.join(line) for line in lines])

    # Display the sentence with new lines every 14 words
    st_image.text((100, 50), f"Your sentence:\n{sentence_lines}", fill="black", font=font)

    # Calculate the height for each sentiment line
    sentiment_height = 30
    sentiment_start_y = 180
    for i, sentiment in enumerate(sorted_sentiments):
        sentiment_y = sentiment_start_y + (i * sentiment_height)
        st_image.text((100, sentiment_y), f"Predicted sentiment: {sentiment}", fill="black", font=font)
 
    # Save the chart as bytes
    chart_bytes = BytesIO()
    fig.savefig(chart_bytes, format='png')
    chart_bytes.seek(0)

    # Open the chart as an image
    chart_image = Image.open(chart_bytes)

    # Combine the text and chart images
    combined_image = Image.new("RGB", (950, 1200), "white")
    combined_image.paste(image, (0, 0))
    combined_image.paste(chart_image, (0, 300))

    # Save the combined image as bytes
    image_bytes = BytesIO()
    combined_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Encode the image bytes to base64
    encoded_image = base64.b64encode(image_bytes.read()).decode()

    # Create the HTML for downloading the image
    href = f'<a href="data:image/png;base64,{encoded_image}" download="sentiment_analysis.png">Download</a>'

    # Display the download link
    st.markdown(href, unsafe_allow_html=True)
     
def show_home():
    st.title("Home")
    st.image("asset/sentiments.png", use_column_width=True)
    st.subheader("Welcome to the Sentiment Analysis App")
    st.write("Explore 5 sentiments: Positive, Moderately Positive, Negative, Moderately Negative, and Neutral. Input text with nuances (e.g., somewhat positive/negative) for an experimental analysis. Observe how the app classifies them, but please note that this is an experiment, and the results may vary. Enjoy your research and learning journey with this exciting exploration! 🚀🔍😊😔.")

    st.write("We value your input! As this application is experimental, your feedback matters. If you spot any issues or have suggestions, click here [Feedback Link](https://forms.microsoft.com/r/qws4Vy7xva) Thank you for helping us improve! 😊")
    
def show_exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    

    df = pd.read_csv('./dataset/sentiment.csv')
    
    st.subheader("Dataset")
    st.write("This dataset is sourced from Kaggle")
    st.dataframe(df[['corpus_name', 'raw_sentence']].sample(10))
    
    st.subheader("Text Cleaning")
    st.dataframe(df[['raw_sentence', 'clean_text']].sample(10))
    
    st.subheader("Sentiment Labeling with TextBlob Analysis")
    st.dataframe(df[['clean_text', 'textblob_polarity', 'sentiment_textblob']].sample(10))
    
        
    # It's important to note that I have decided to save the processed images to avoid the need for repetitive processing,
    # which can be time-consuming. This approach allows for quicker access to the results and facilitates further analysis
    
    st.subheader("Sentiment Anayisis: Unbalanced Data vs Balanced Data")
    st.image("asset/new_sentiment_analysis.png", use_column_width=True)
    
    st.subheader("Top 20 Frequently used Words")
    st.image("asset/new_most_frequently.png", use_column_width=True)
    
    st.subheader("Random Samples with Positive Sentiment")
    positive_samples = df[df['sentiment_textblob'] == 'Positive'][['clean_text', 'sentiment_textblob']].sample(10)
    st.dataframe(positive_samples)
    
    st.subheader("Random Samples with Negative Sentiment")
    positive_samples = df[df['sentiment_textblob'] == 'Negative'][['clean_text', 'sentiment_textblob']].sample(10)
    st.dataframe(positive_samples)
    
    st.subheader("Positive Word cloud")
    st.image("asset/new_positive.png", use_column_width=True)
    
    st.subheader("Moderately Positive Word cloud")
    st.image("asset/new_moderately_positive.png", use_column_width=True)
    
    st.subheader("Neutral Word cloud")
    st.image("asset/new_neutral.png", use_column_width=True)
    
    st.subheader("Moderately Negative Word cloud")
    st.image("asset/new_moderately_negative.png", use_column_width=True)
    
    st.subheader("Negative Word cloud")
    st.image("asset/new_negative.png", use_column_width=True)
    
    st.subheader("Model Evaluation: Performance Metrics Comparison")
    st.image("asset/new_classifier.png", use_column_width=True)

    st.subheader("Logistic Regression Classifier Performance Metrics")
    st.image("asset/new_evaluation.png", use_column_width=True)
    
    st.subheader("Confusion Matrix")
    st.image("asset/new_confusion_matrix.png", use_column_width=True)    

def show_author():
    st.title("Author")
    col1, col2 = st.columns(2)  # Split the layout into two columns

    # First Column
    with col1:
        st.image("asset/profile.png", width=200)
        st.markdown('<a href="https://www.linkedin.com/in/christiandls444/" target="_blank">LinkedIn Profile</a>', unsafe_allow_html=True)

    # Second Column
    with col2:
        st.write("Hey everyone! I'm Christian M. De Los Santos, from the Philippines. I have over 2 years of experience in the field of data analytics, with a special focus on machine learning. I firmly believe that AI and ML have the power to bring about positive change in our communities, which is why I'm here, eager to make an impact. Learning from all of you brilliant minds is something I'm truly looking forward to. Let's collaborate and create something amazing together!")


def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Sentiment Detection", "Exploratory Data Analysis", "Author"])

    if selected_page == "Home":
        show_home()
    elif selected_page == "Sentiment Detection":
        show_sentiment_detection()
    elif selected_page == "Exploratory Data Analysis":
        show_exploratory_data_analysis()
    elif selected_page == "Author":
        show_author()


if __name__ == "__main__":
    main()