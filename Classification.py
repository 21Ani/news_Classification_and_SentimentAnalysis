import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving the model

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
# Load dataset
dataset = pd.read_csv("labelled.csv")

# Create CategoryId from the Category column
dataset['CategoryId'] = dataset['Category'].factorize()[0]

# Create a mapping from CategoryId to Category
category_mapping = dict(zip(dataset['CategoryId'], dataset['Category']))

# Preprocessing functions
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = ''.join([x if x.isalnum() else ' ' for x in text])  # Remove special characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in words])  # Lemmatization

# Preprocess the 'Body' column
dataset['Body'] = dataset['Body'].apply(preprocess_text)

# Prepare data for modeling
x = np.array(dataset['Body'])
y = np.array(dataset['CategoryId'])

# Vectorize the text
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(x).toarray()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Support Vector Classifier': SVC(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'K Nearest Neighbour': KNeighborsClassifier(n_neighbors=10),
}

# Train models and save them
trained_models = {}
for model_name, model in models.items():
    oneVsRest = OneVsRestClassifier(model)
    oneVsRest.fit(x_train, y_train)
    trained_models[model_name] = oneVsRest
    # Save each model
    # joblib.dump(oneVsRest, f"{model_name.replace(' ', '_')}.joblib")

# Save the CountVectorizer
# joblib.dump(cv, "count_vectorizer.joblib")


# Function to preprocess user input and predict
def predict_category(user_input):
    user_input = preprocess_text(user_input)
    user_input_vector = cv.transform([user_input]).toarray()

    predictions = {}
    for model_name in trained_models.keys():
        pred = trained_models[model_name].predict(user_input_vector)[0]
        category_name = category_mapping.get(pred, "Unknown")
        predictions[model_name] = category_name

    return predictions

if __name__ == "__main__":
    while True:
        user_input = input("Enter the text you want to classify (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        predictions = predict_category(user_input)

        # Display predictions
        for model_name, category_name in predictions.items():
            print(f"{model_name} predicted category: {category_name}")

