import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

class SentimentAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Sentiment Analysis")

        self.label = tk.Label(master, text="Sentiment Analysis of Social Media Data")
        self.label.pack()

        self.load_button = tk.Button(master, text="Load Data", command=self.load_data)
        self.load_button.pack()

        self.clean_button = tk.Button(master, text="Clean Data", command=self.clean_data)
        self.clean_button.pack()

        self.plot_sentiment_button = tk.Button(master, text="Plot Sentiment Distribution", command=self.plot_sentiment_distribution)
        self.plot_sentiment_button.pack()

        self.wordcloud_button = tk.Button(master, text="Generate Word Cloud", command=self.generate_wordcloud)
        self.wordcloud_button.pack()

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.evaluate_button = tk.Button(master, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.pack()

        self.training_data = None
        self.validation_data = None
        self.model = None
        self.vectorizer = None

    def load_data(self):
        training_data_path = filedialog.askopenfilename(title="Select Training Data CSV")
        validation_data_path = filedialog.askopenfilename(title="Select Validation Data CSV")

        self.training_data = pd.read_csv(training_data_path)
        self.validation_data = pd.read_csv(validation_data_path)

        print("Training Data Columns:", self.training_data.columns.tolist())
        print("Validation Data Columns:", self.validation_data.columns.tolist())

        messagebox.showinfo("Load Data", "Data loaded successfully!")

    def clean_data(self):
        if self.training_data is None or self.validation_data is None:
            messagebox.showerror("Error", "Load data first!")
            return

        text_column_train = self.training_data.columns[3]
        text_column_val = self.validation_data.columns[3]

        self.training_data = self._clean_data(self.training_data, text_column_train)
        self.validation_data = self._clean_data(self.validation_data, text_column_val)

        messagebox.showinfo("Clean Data", "Data cleaned successfully!")

    def _clean_data(self, df, text_column):
        df = df.drop_duplicates()
        df = df.dropna(subset=[text_column])
        return df

    def plot_sentiment_distribution(self):
        if self.training_data is None:
            messagebox.showerror("Error", "Load data first!")
            return

        sentiment_column_train = 'Positive'

        plt.figure(figsize=(8, 6))
        sns.countplot(x=sentiment_column_train, data=self.training_data)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

    def generate_wordcloud(self):
        if self.training_data is None:
            messagebox.showerror("Error", "Load data first!")
            return

        text_column_train = self.training_data.columns[3]

        text = " ".join(tweet for tweet in self.training_data[text_column_train])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def train_model(self):
        if self.training_data is None or self.validation_data is None:
            messagebox.showerror("Error", "Load data first!")
            return

        text_column_train = self.training_data.columns[3]
        text_column_val = self.validation_data.columns[3]

        self.training_data['cleaned_text'] = self.training_data[text_column_train].apply(self._preprocess_text)
        self.validation_data['cleaned_text'] = self.validation_data[text_column_val].apply(self._preprocess_text)

        self.vectorizer = CountVectorizer(stop_words='english')
        X_train = self.vectorizer.fit_transform(self.training_data['cleaned_text'])
        X_validation = self.vectorizer.transform(self.validation_data['cleaned_text'])

        sentiment_column_train = 'Positive'
        y_train = self.training_data[sentiment_column_train]

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

        messagebox.showinfo("Train Model", "Model trained successfully!")

    def evaluate_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Train model first!")
            return

        sentiment_column_val = 'Irrelevant'
        y_validation = self.validation_data[sentiment_column_val]

        X_validation = self.vectorizer.transform(self.validation_data['cleaned_text'])
        y_pred = self.model.predict(X_validation)

        print(classification_report(y_validation, y_pred))
        cm = confusion_matrix(y_validation, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def _preprocess_text(self, text):
        text = text.lower()
        text = "".join([char for char in text if char.isalpha() or char.isspace()])
        return text


root = tk.Tk()
app = SentimentAnalysisApp(root)
root.mainloop()
