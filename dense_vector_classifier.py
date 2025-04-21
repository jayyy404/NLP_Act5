import wikipedia
import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils.multiclass import unique_labels


#The reason for this is I need to have more data to train the model
topics = {
    "Harry Potter": ["Harry Potter", "Hogwarts", "J.K. Rowling"],
    "Game of Thrones": ["Game of Thrones", "House Stark", "Westeros"],
    "COVID-19": ["COVID-19", "COVID-19 pandemic", "COVID-19 vaccine"],
    "Artificial Intelligence": ["Artificial Intelligence", "AI ethics", "AI applications"],
    "Machine Learning": ["Machine Learning", "Deep learning", "Supervised learning"]
}

documents = []
labels = []


for idx, topic in enumerate(topics):
    try:
        summary = wikipedia.summary(topic, sentences=2, auto_suggest=False)
        documents.append(summary.lower().split())  
        labels.append(topic)  
        print(f"Fetched: {topic}")
    except:
        print(f"Failed to fetch: {topic}")
        documents.append([])
        labels.append(topic)  


w2v_model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=2, seed=42)


def document_vector(doc):
    vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.array([document_vector(doc) for doc in documents])
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


topic_labels_present = list(unique_labels(y_test, y_pred))

print(classification_report(y_test, y_pred, labels=topic_labels_present, target_names=topic_labels_present))
