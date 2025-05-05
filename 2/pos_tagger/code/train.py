from collections import defaultdict
import math
import pickle

START_TOKEN = "<s>" # The special token used for marking the beginning of the sentences

def train_hmm(train_file="POS_train.pos", model_file="hmm_model.pkl"):
  emission_counts = defaultdict(lambda: defaultdict(int))  # tag, word, count
  transition_counts = defaultdict(lambda: defaultdict(int))  # prev_tag, tag, count
  tag_counts = defaultdict(int)
  start_counts = defaultdict(int)
  known_words = set()

  prev_tag = START_TOKEN # Initialize with sentence start

  with open(train_file, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()

      if not line:
        prev_tag = START_TOKEN # Reset at boundary
        continue

      if '\t' not in line:
        continue # Skip bad lines

      word, tag = line.split('\t')
      known_words.add(word)

      # Count the emissions: P(word | tag)
      emission_counts[tag][word] += 1
      tag_counts[tag] += 1

      # Count start tags separately
      if prev_tag == START_TOKEN:
        start_counts[tag] += 1
      else:
        # Count transitions: P(tag_i | tag_{i-1})
        transition_counts[prev_tag][tag] += 1

      prev_tag = tag

  # Compute log probabilities with Laplace smoothing
  tags = list(tag_counts.keys())
  vocabulary = list(known_words)

  # Emission Probabilities: log P(word | tag) 
  emission_probs = {}
  for tag in tags:
    total = tag_counts[tag] + len(vocabulary)
    emission_probs[tag] = {}
    for word in emission_counts[tag]:
      prob = (emission_counts[tag][word] + 1) / total
      emission_probs[tag][word] = math.log(prob)

  # Transition Probabilities: log P(tag_i | tag_{i-1})
  transition_probs = {}
  for prev in tags + [START_TOKEN]:
    total = sum(transition_counts[prev].values()) + len(tags)
    transition_probs[prev] = {}
    for tag in tags:
      prob = (transition_counts[prev][tag] + 1) / total
      transition_probs[prev][tag] = math.log(prob)

  # Start Probabilities: log P(tag | <s>)
  start_probs = {}
  total_starts = sum(start_counts.values()) + len(tags)
  for tag in tags:
    prob = (start_counts[tag] + 1) / total_starts
    start_probs[tag] = math.log(prob)

  # Save model as a pickled dictionary 
  with open(model_file, "wb") as f:
    pickle.dump({
      "tags": tags,
      "known_words": known_words,
      "emission_probs": emission_probs,
      "transition_probs": transition_probs,
      "start_probs": start_probs
    }, f)

  print("Model training finished. Saved to", model_file)

if __name__ == "__main__":
  train_hmm()
