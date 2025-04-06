import pandas as pd
import ssl
from striprtf.striprtf import rtf_to_text
import re
import string
from sklearn.model_selection import train_test_split
import numpy as np

def squinky_dataset(url):
  ssl._create_default_https_context = ssl._create_unverified_context

  squinky_data = pd.read_csv(url)
  squinky_data = squinky_data[['formality', 'sentence']]
  squinky_data['class'] =[0 if x > 4.0 else 1 for x in squinky_data['formality']]
  squinky_data = squinky_data.drop('formality', axis=1)

  return squinky_data

def gpt_dataset(filenames):
  formal_sentences = []
  informal_sentences = []

  for filename in filenames:
      with open(filename, 'r', encoding='utf-8') as file:
          rtf_content = file.read()

          plain_text = rtf_to_text(rtf_content)

          plain_text = plain_text.split('\n')

          sentences = [x for x in plain_text if x != '']

          if filename == 'formal_sentences.rtf':
              formal_sentences = sentences
          elif filename == 'informal_sentences.rtf':
              informal_sentences = sentences
          elif filename == 'middle_sentences.rtf':
              for sentence in sentences:
                class_ = int(sentence[-2:])
                if class_ == 0:
                  formal_sentences.append(sentence[:-2])
                else:
                  informal_sentences.append(sentence[:-2])

  gpt_data = [formal_sentences + informal_sentences, [0] * len(formal_sentences) + [1] * len(informal_sentences)]

  gpt_data = pd.DataFrame(gpt_data, index=['sentence', 'class']).T

  return gpt_data

def get_all_data(squinky_url, gpt_filenames):
  squinky_data = squinky_dataset(squinky_url)
  gpt_data = gpt_dataset(gpt_filenames)

  all_data = pd.concat([squinky_data, gpt_data], ignore_index=True)

  return all_data

def extract_features(text):

  slang_words = {
      "gonna", "wanna", "gotta", "bro", "bruh", "lol", "lmao", "rofl", "yo", "wassup", "lit", "fam", "dude", "nah", "yolo",
      "omg", "idk", "smh", "tbh", "btw", "thx", "pls", "fyi", "imo", "imho", "nope", "yup", "dope", "sick", "chill", "brb",
      "ttyl", "bff", "bae", "hmu", "jk", "np", "srsly", "ngl", "ikr", "cuz", "coz", "sup", "yo", "meh", "okie"
  }

  signoffs = {
      "regards", "sincerely", "best", "best regards", "kind regards", "warm regards", "respectfully", "cheers", "thanks",
      "thank you", "yours truly", "yours sincerely", "warmest regards", "take care", "all the best", "many thanks",
      "appreciatively", "with appreciation", "gratefully", "later", "see ya", "peace out"
  }

  polite_words = {
      "please", "kindly", "would you", "could you", "may I", "if you don't mind", "would it be possible", "I'd appreciate it",
      "thank you", "thanks in advance", "pardon me", "excuse me", "would you mind", "it would be great if", "I would be grateful",
      "may I request", "your assistance is appreciated", "if possible"
  }

  formal_words = {
      "regarding", "therefore", "furthermore", "moreover", "accordingly", "respectfully", "hence", "nevertheless",
      "thus", "notwithstanding", "consequently", "in accordance with", "pursuant to", "aforementioned", "hereinafter",
      "henceforth", "insofar", "nonetheless", "in the event that", "in lieu of", "whereas", "wherein", "inasmuch as",
      "prior to", "subsequently", "to that end", "with reference to"
  }

  contraction_pattern = re.compile(r"\b(?:[A-Za-z]+n't|[A-Za-z]+'ll|[A-Za-z]+'ve|[A-Za-z]+'re|[A-Za-z]+'d|[A-Za-z]+'m)\b")

  words = text.split()
  num_words = len(words) # amount of words
  num_chars = len(text) # amount of symbols
  num_upper = sum(1 for c in text if c.isupper()) # capital letters
  num_exclam = text.count('!') # exclamation mark amount
  num_contractions = len(contraction_pattern.findall(text)) # contraction amount
  num_slang = sum(1 for word in words if word.lower() in slang_words) # amount of slang words
  num_signoffs = sum(1 for word in words if word.lower() in signoffs) # amount of signoffs
  num_polite = sum(1 for word in words if word.lower() in polite_words) # amount of polite words
  num_formal = sum(1 for word in words if word.lower() in formal_words) # amount of formal words
  num_punct = sum(1 for c in text if c in string.punctuation) # number of punctuation chars
  digit_ratio = sum(c.isdigit() for c in text) / (num_chars + 1e-5) # if the text is heavy on digits

  return [
      num_chars,
      np.mean([len(word) for word in words]) if words else 0,
      num_upper,
      num_exclam,
      num_contractions,
      num_slang,
      num_signoffs,
      num_polite,
      num_formal,
      num_punct,
      digit_ratio
  ]

def catboost_data_preparation(all_data):

  feature_names = [
      "text_length", "avg_word_length", "uppercase_count", "exclamation_count",
      "contraction_count", "slang_count", "signoff_count", "polite_count",
      "formal_count", "punctuation_count", "digit_ratio"
  ]

  all_data[feature_names] = all_data["sentence"].apply(lambda x: pd.Series(extract_features(x)))

  X = all_data[feature_names]
  y = all_data["class"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

  return X_train, X_test, y_train, y_test





