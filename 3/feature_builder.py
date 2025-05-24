import sys
import re

# Load the gazetteers
with open("LargestCity.txt") as f:
  city_list = set(line.strip().lower() for line in f if line.strip())

with open("dist.all.last.txt") as f:
  name_list = set(line.strip().lower() for line in f if line.strip())


def word_shape(word):
  shape = ""
  for c in word:
    if c.isupper():
      shape += "X"
    elif c.islower():
      shape += "x"
    elif c.isdigit():
      shape += "d"
    else:
      shape += c
  return shape

def extract_features(tokens, pos_tags, chunk_tags, prev_tags=None, is_training=True):
  features_per_token = []

  for i in range(len(tokens)):
    features = []
    word = tokens[i]

    # Gazetteer features
    if word.lower() in city_list:
      features.append("in_city_list")
    if word.lower() in name_list:
      features.append("in_name_list")

    # Basic
    features.append(f"word={word}")
    features.append(f"pos={pos_tags[i]}")
    features.append(f"chunk={chunk_tags[i]}")

    # Capitalization + digit
    if word.isupper():
      features.append("isupper")
    if word.istitle():
      features.append("istitle")
    if word.isdigit():
      features.append("isdigit")

    # Prefix/suffix
    if len(word) >= 2:
      features.append(f"suffix2={word[-2:]}")
    if len(word) >= 3:
      features.append(f"suffix3={word[-3:]}")

    # Word shape
    features.append(f"shape={word_shape(word)}")

    # Context
    if i > 0:
      features.append(f"prev_word={tokens[i - 1]}")
      features.append(f"prev_pos={pos_tags[i - 1]}")
    else:
      features.append("BOS")

    if i < len(tokens) - 1:
      features.append(f"next_word={tokens[i + 1]}")
      features.append(f"next_pos={pos_tags[i + 1]}")
    else:
      features.append("EOS")

    # Previous tag
    if is_training:
      features.append(f"prev_tag={prev_tags[i]}")
    else:
      features.append("prev_tag=@@")

    features_per_token.append(features)

  return features_per_token

def read_data(filename, with_labels=True):
  sentences = []
  sentence = []
  with open(filename) as f:
    for line in f:
      line = line.strip()
      if not line:
        if sentence:
          sentences.append(sentence)
          sentence = []
        continue
      parts = line.split("\t")
      sentence.append(parts)
    if sentence:
      sentences.append(sentence)
  return sentences

def build_feature_file(input_path, output_path, is_training=True):
  data = read_data(input_path, with_labels=is_training)

  with open(output_path, "w") as out:
    for sentence in data:
      tokens = [t[0] for t in sentence]
      pos_tags = [t[1] for t in sentence]
      chunk_tags = [t[2] for t in sentence]
      name_tags = [t[3] if is_training and len(t) > 3 else None for t in sentence]

      prev_tags = ["BOS"] + name_tags[:-1] if is_training else ["@@"] * len(tokens)
      features = extract_features(tokens, pos_tags, chunk_tags, prev_tags, is_training)

      for i in range(len(tokens)):
        line = tokens[i] + "\t" + "\t".join(features[i])
        if is_training:
          line += "\t" + name_tags[i]
        out.write(line + "\n")
      out.write("\n")

if __name__ == "__main__":
  input_file = sys.argv[1]
  output_file = sys.argv[2]
  is_training = sys.argv[3].lower() == "true"
  build_feature_file(input_file, output_file, is_training)
