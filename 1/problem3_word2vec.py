import nltk
nltk.download('brown')

from nltk.corpus import brown
from gensim.models import Word2Vec

from gensim.models import KeyedVectors

# Google News pre-trained model
print("Loading Google News Word2Vec model...")
google_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
print("Model loaded.")

# Similar to "rebellion" and "slave"
words1_google = google_model.most_similar("rebellion", topn=10)
words2_google = google_model.most_similar("slave", topn=10)

print("\n[GoogleNews] Similar to 'rebellion':")
for word, score in words1_google:
    print(f"{word}: {score:.4f}")

print("\n[GoogleNews] Similar to 'slave':")
for word, score in words2_google:
    print(f"{word}: {score:.4f}")


# Load Brown corpus as list of sentences
sentences = brown.sents()

# Train Word2Vec with CBOW model, 100 dimensions, window=5, and min word count=2
model_brown = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=0)

# Save the model
model_brown.save("brown_word2vec.model")

# Check word existence and get most similar words
def get_similar_words(word):
  if word in model_brown.wv.key_to_index:
    return model_brown.wv.most_similar(word, topn=10)
  else:
    return f"'{word}' not in vocabulary."

# Query
targets = ["rebellion", "slave"]
for target in targets:
  print(f"\nSimilar to '{target}':")
  result = get_similar_words(target)
  if isinstance(result, str):
    print(result)
  else:
    for word, score in result:
      print(f"{word}: {score:.4f}")
