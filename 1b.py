import math
import wikipedia
from collections import Counter

# Topics
topics = ["Harry Potter", "Game of Thrones", "COVID-19", "Artificial Intelligence", "Machine Learning"]
documents = []


for topic in topics:
    try:
        summary = wikipedia.summary(topic, sentences=2, auto_suggest=False)
        documents.append(summary)
        print(f"Fetched summary for: {topic}")
    except wikipedia.exceptions.PageError:
        print(f"Page not found: {topic}")
        documents.append("")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Error for '{topic}', using first suggestion: {e.options[0]}")
        try:
            summary = wikipedia.summary(e.options[0], sentences=2, auto_suggest=False)
            documents.append(summary)
        except:
            documents.append("")
    except:
        documents.append("")


tokenized_docs = [doc.lower().split() for doc in documents]


vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

# Compute Term Frequency (TF)
def compute_tf(doc_tokens, vocabulary):
    word_count = Counter(doc_tokens)
    return {term: word_count.get(term, 0) for term in vocabulary}

tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

# Compute Inverse Document Frequency (IDF)
def compute_idf(tokenized_docs, vocabulary):
    idf_dict = {}
    total_docs = len(tokenized_docs)
    for term in vocabulary:
        doc_count = sum(1 for doc in tokenized_docs if term in doc)
        idf_dict[term] = math.log((total_docs / (1 + doc_count))) + 1  
    return idf_dict

idf = compute_idf(tokenized_docs, vocabulary)

# Compute TF-IDF
def compute_tfidf(tf_vector, idf, vocabulary):
    return {term: tf_vector[term] * idf[term] for term in vocabulary}

tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]


print("\nTerm-Document Matrix (TF-IDF):")
print("Term".ljust(20), end="")
for i in range(len(documents)):
    print(f"Doc{i+1}".rjust(10), end="")
print()

for term in vocabulary:
    print(term.ljust(20), end="")
    for vec in tfidf_vectors:
        print(f"{vec[term]:10.2f}", end="")
    print()
