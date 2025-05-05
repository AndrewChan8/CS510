import pickle
import math
import sys
from collections import defaultdict

UNKNOWN_PENALTY = -5.0 # Fallback log prob for unknowns

# Load the trained pickle model
def load_model(model_file="hmm_model.pkl"):
  with open(model_file, "rb") as f:
    return pickle.load(f)

# Read into the file into a list of sentences
def read_input_file(word_file):
  sentences = []
  current = []

  with open(word_file, "r", encoding="utf-8") as f:
    for line in f:
      word = line.strip()
      if word == "":
        if current:
          sentences.append(current)
          current = []
      else:
        current.append(word)
    if current:
      sentences.append(current)
  return sentences

# Write the tagged output to a .pos file
def write_output_file(sentences, tags, output_file):
  with open(output_file, "w", encoding="utf-8") as f:
    for sentence, tag_seq in zip(sentences, tags):
      for word, tag in zip(sentence, tag_seq):
        f.write(f"{word}\t{tag}\n")
      f.write("\n")

def get_emission_logprob(emission_probs, tag, word, known_words):
  if word in emission_probs.get(tag, {}):
    return emission_probs[tag][word] # Known word, but not emitted by this tag
  elif word in known_words:
    return math.log(1e-10) # Known word, but not emitted by this tag
  else:
    # Unknown word heuristics
    if word.istitle():
      if tag == "NNP": return -1.0
    elif word.isdigit():
      if tag == "CD": return -1.0
    elif word.endswith("ing"):
      if tag == "VBG": return -1.0
    elif word.endswith("ed"):
      if tag == "VBD": return -1.0
    elif word.endswith("ly"):
      if tag == "RB": return -1.0
    elif word.endswith("ion"):
      if tag == "NN": return -1.0
    elif word.endswith("s") and not word.istitle():
      if tag == "NNS": return -1.0
    elif word.endswith("est"):
      if tag == "JJS": return -1.0
    elif word.endswith("er"):
      if tag in ("JJR", "NN"): return -1.0
    elif "-" in word:
      if tag in ("JJ", "NN"): return -1.0
    elif word.startswith("$") or word.endswith("%"):
      if tag == "CD": return -1.0

    if tag in ("NN", "VB", "JJ"):
      return -3.0  # Gentle boost to common tags
    return UNKNOWN_PENALTY  # Fallback for others

def viterbi(sentence, model):
  tags = model["tags"]
  known_words = model["known_words"]
  emission_probs = model["emission_probs"]
  transition_probs = model["transition_probs"]
  start_probs = model["start_probs"]

  # V[t][tag] stores the max log-prob of a path ending in tag at time
  V = [{} for _ in range(len(sentence))]
  backpointer = [{} for _ in range(len(sentence))]

  # Init first word
  for tag in tags:
    trans = start_probs.get(tag, math.log(1e-10))
    emiss = get_emission_logprob(emission_probs, tag, sentence[0], known_words)
    V[0][tag] = trans + emiss
    backpointer[0][tag] = None

  # Recursion
  for t in range(1, len(sentence)):
    word = sentence[t]
    for curr_tag in tags:
      max_prob = float("-inf")
      best_prev = None
      for prev_tag in tags:
        trans = transition_probs.get(prev_tag, {}).get(curr_tag, math.log(1e-10))
        emiss = get_emission_logprob(emission_probs, curr_tag, word, known_words)
        prob = V[t - 1][prev_tag] + trans + emiss
        if prob > max_prob:
          max_prob = prob
          best_prev = prev_tag
      V[t][curr_tag] = max_prob
      backpointer[t][curr_tag] = best_prev

  # Termination: backtrace best tag sequence
  last_probs = V[-1]
  last_tag = max(last_probs, key=last_probs.get)
  tags_sequence = [last_tag]

  for t in reversed(range(1, len(sentence))):
    last_tag = backpointer[t][last_tag]
    tags_sequence.insert(0, last_tag)

  return tags_sequence

def decode(input_file="POS_dev.words", output_file="dev_output.pos", model_file="hmm_model.pkl"):
  model = load_model(model_file)
  sentences = read_input_file(input_file)
  predicted_tags = []

  for sentence in sentences:
    tag_seq = viterbi(sentence, model)
    predicted_tags.append(tag_seq)

  write_output_file(sentences, predicted_tags, output_file)
  print(f"Finished decoding. Output stored in {output_file}")

if __name__ == "__main__":
  import sys
  if len(sys.argv) == 3:
    decode(input_file=sys.argv[1], output_file=sys.argv[2])
  else:
    decode()  # defaults to dev set
