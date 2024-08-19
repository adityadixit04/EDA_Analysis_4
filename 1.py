import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

training_data_path = "D:\\Study\\Intership\\Task 4\\twitter_training.csv"
validation_data_path = "D:\\Study\\Intership\\Task 4\\twitter_validation.csv"

training_data = pd.read_csv(training_data_path)
validation_data = pd.read_csv(validation_data_path)

print("Training Data Columns:", training_data.columns.tolist())
print("Validation Data Columns:", validation_data.columns.tolist())

sentiment_column_train = 'Positive' 
sentiment_column_val = 'Irrelevant'  
text_column_train = training_data.columns[3]  
text_column_val = validation_data.columns[3]  

def clean_data(df, text_column):
    df = df.drop_duplicates()
    df = df.dropna(subset=[text_column])
    return df

training_data = clean_data(training_data, text_column_train)
validation_data = clean_data(validation_data, text_column_val)

def plot_sentiment_distribution(df, sentiment_column):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=sentiment_column, data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

plot_sentiment_distribution(training_data, sentiment_column_train)

def generate_wordcloud(df, text_column):
    text = " ".join(tweet for tweet in df[text_column])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

generate_wordcloud(training_data, text_column_train)

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char.isalpha() or char.isspace()])
    return text

training_data['cleaned_text'] = training_data[text_column_train].apply(preprocess_text)
validation_data['cleaned_text'] = validation_data[text_column_val].apply(preprocess_text)

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(training_data['cleaned_text'])
X_validation = vectorizer.transform(validation_data['cleaned_text'])

y_train = training_data[sentiment_column_train]
y_validation = validation_data[sentiment_column_val]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)

def evaluate_model(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model(y_validation, y_pred)

def visualize_sentiment_patterns(df):
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.split())
    df_exploded = df.explode('cleaned_text')
    top_words = df_exploded['cleaned_text'].value_counts().head(20)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_words.values, y=top_words.index, palette='viridis')
    plt.title('Top Words in Tweets')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()

visualize_sentiment_patterns(training_data)
