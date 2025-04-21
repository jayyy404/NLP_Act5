import wikipedia
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt')

#The reason for this is i need more data to train the model.
topics = {
    "Harry Potter": ["Harry Potter", "Hogwarts", "J.K. Rowling"],
    "Game of Thrones": ["Game of Thrones", "House Stark", "Westeros"],
    "COVID-19": ["COVID-19", "COVID-19 pandemic", "COVID-19 vaccine"],
    "Artificial Intelligence": ["Artificial Intelligence", "AI ethics", "AI applications"],
    "Machine Learning": ["Machine Learning", "Deep learning", "Supervised learning"]
}


documents = []
labels = []

for topic_name, search_terms in topics.items():
    for term in search_terms:
        try:
            summary = wikipedia.summary(term, sentences=10) #ari sir gin set ko ang sentences nga kwaon nga 10 sentences. If small sentences or data mas better ang TF-IDF
            sentences = sent_tokenize(summary)
            for sent in sentences:
                documents.append(sent.lower().split())
                labels.append(topic_name)
        except:
            continue 


model = Word2Vec(
    sentences=documents,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)


def doc_to_vector(doc):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.array([doc_to_vector(doc) for doc in documents])
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train Logistic Regression
clf = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced'
)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))