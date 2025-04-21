import wikipedia
from collections import Counter

# Topics
topics = ["Harry Potter", "Game of Thrones", "COVID-19", "Artificial Intelligence", "Machine Learning"]
documents = []

# Fetch Wikipedia summaries safely
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
        except Exception as err:
            print(f"Could not fetch fallback for '{e.options[0]}': {err}")
            documents.append("")
    except Exception as e:
        print(f"Unknown error for '{topic}': {e}")
        documents.append("")

# Tokenize
tokenized_docs = [doc.lower().split() for doc in documents]

# Build vocabulary
vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

# Compute raw term frequency
def compute_tf(doc_tokens, vocabulary):
    word_count = Counter(doc_tokens)
    return {term: word_count.get(term, 0) for term in vocabulary}

tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

# Display matrix
print("\nTerm-Document Matrix (Raw Frequency):")
print("Term".ljust(20), end="")
for i in range(len(documents)):
    print(f"Doc{i+1}".rjust(8), end="")
print()

for term in vocabulary:
    print(term.ljust(20), end="")
    for doc_tf in tf_vectors:
        print(f"{doc_tf[term]:>8}", end="")
    print()
