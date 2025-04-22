from flask import Flask, render_template, request
import sqlite3, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import pandas as pd

# Load the CSV data once when app starts
csv_file_path = "people-100.csv"  # relative path
csv_data = pd.read_csv(csv_file_path)

# Preview in terminal to verify it's working
print("CSV Loaded Successfully:")
print(csv_data.head())



app = Flask(__name__)
DATABASE = 'projects.db'

# DB create function
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    abstract TEXT
                )''')
    conn.commit()
    conn.close()

# Insert data
def add_project(title, abstract):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO projects (title, abstract) VALUES (?, ?)", (title, abstract))
    conn.commit()
    conn.close()

# Get all old projects
def get_all_projects():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT title, abstract FROM projects")
    data = c.fetchall()
    conn.close()
    return data

# Check similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize necessary objects for NLP
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.stem import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization (split into words)
    words = text.split()
    
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in ENGLISH_STOP_WORDS]
    
    # Join words back into a single string
    return ' '.join(words)


def get_all_projects():
    # Database connection
    conn = sqlite3.connect('projects.db')
    c = conn.cursor()
    c.execute("SELECT title, abstract FROM projects")
    projects = c.fetchall()
    conn.close()
    
    return projects

def get_all_projects():
    # Database connection
    conn = sqlite3.connect('projects.db')
    c = conn.cursor()
    c.execute("SELECT title, abstract FROM projects")
    projects = c.fetchall()
    conn.close()
    
    if not projects:
        print("No projects found in the database.")
    return projects

def check_similarity(new_title, new_abstract):
    # Preprocess the new inputs
    new_title = preprocess_text(new_title)
    new_abstract = preprocess_text(new_abstract)
    
    # Get all existing projects
    projects = get_all_projects()
    
    if not projects:
        return None  # Returning None if no projects in the database

    # Preprocess existing projects
    existing_projects = [(preprocess_text(title), preprocess_text(abstract)) for title, abstract in projects]
    
    # Combine the new input with existing projects
    texts = [new_title + " " + new_abstract] + [title + " " + abstract for title, abstract in existing_projects]
    
    # Check if texts list is empty
    if not texts:
        print("No valid texts to compare!")
        return None  # Return None if there's no text to compare
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(tfidf[0:1], tfidf[1:])
    
    # Calculate similarity percentage
    similarity = sim_matrix[0][0] * 100
    matched_project = projects[sim_matrix.argmax()][0]  # Get the matched project title
    
    return round(similarity, 2), matched_project





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    title = request.form['title']
    abstract = request.form['abstract']
    similarity, matched_project = check_similarity(title, abstract)

    if similarity >= 60:
        return render_template('result.html', status="duplicate", similarity=round(similarity, 2), match=matched_project)
    else:
        add_project(title, abstract)
        return render_template('result.html', status="unique", similarity=round(similarity, 2))

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)

def add_dummy_data():
    dummy_projects = [
        ("AI in Healthcare", "Using AI for disease prediction and diagnosis."),
        ("Smart Traffic System", "IoT-based traffic lights with real-time flow analysis."),
        ("Face Recognition Attendance", "Using computer vision for marking attendance."),
        ("E-Voting System", "Blockchain based voting platform."),
        ("Solar Energy Tracker", "Automatic adjustment of solar panels using sensors.")
    ]
    for title, abstract in dummy_projects:
        add_project(title, abstract)

# Call it once
add_dummy_data()
