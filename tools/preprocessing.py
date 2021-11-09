from __future__ import annotations

import os

import pandas as pd
from nltk import pos_tag

# You can comment out the below after you've run the code once. It saves
# files the nltk library needs in an nltk specific directory in your home
# directory. If you don't like having that directory there, you can just
# delete it after you've used the nltk library to run some code and it won't
# cause any problems.
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
# nltk.download('punkt')


def load_and_prep_files(file_names: list[str],
                        directory: str = '') -> \
        tuple[list[str], list[str], dict[str, int]]:

    raw_text = ''

    for file_name in file_names:
        print(f"Loading {file_name}...")
        path = os.path.join(directory, file_name)
        with open(path, 'r') as file:
            raw_text += ' ' + file.read() + ' '
    print("All text files loaded.")
    print("Beginning file preprocessing...")

    # Convert multiples spaces, tabs, newlines, etc to single space
    text = ' '.join(raw_text.split())

    # Remove punctuation
    punctuation = [
        '~', '`', '!', '@', '#', '$', '%', '^', '&', '*',
        '(', ')', '-', '_', '+', '=', '{', '[', '}', ']',
        '|', '\\', ':', ';', '"', "'", '<', ',', '>', '.',
        '?', '/', "’", '–', '‘', '“', '”', '—'
    ]
    for char in punctuation:
        text = text.replace(char, '')

    # Turns out that that introduces some new multiple white spaces:
    text = ' '.join(text.split())

    # Remove numbers:
    numbers = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
    ]
    for number in numbers:
        text = text.replace(number, '')

    # Make it all lower case
    text = text.lower()
    # Split into list of strings
    text = text.split()
    print(f"File preprocessing complete.")

    print(f"Text Length = {len(text)}")

    vocabulary = list(set(text))

    print(f"Vocabulary Length = {len(vocabulary)}")

    word_counts = {
        word: text.count(word) for word in vocabulary
    }

    return text, vocabulary, word_counts


def print_words_with_counts(word_counts: dict[str, int],
                            reverse: bool = True,
                            save_to_file: str | False = False):
    paired_data = sorted(
        word_counts.items(), reverse=reverse, key=lambda pair: pair[1]
    )
    for word, count in paired_data:
        print(f"{word}: {count}")
    if save_to_file:
        with open(save_to_file, 'w') as file:
            df = pd.DataFrame.from_dict(word_counts)
            df.to_csv(file, index=False, header=True)
    print(f"Data saved to '{save_to_file}'")


def tag_w_pos_nlp(text: list[str]) -> list[tuple[str, str]]:
    tokenized_text = pos_tag(text)
    return tokenized_text
