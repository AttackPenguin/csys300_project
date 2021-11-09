from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
import os

# You can comment out the below after you've run the code once. It saves
# files the nltk library needs in an nltk specific directory in your home
# directory. If you don't like having that directory there, you can just
# delete it after you've used the nltk library to run some code and it won't
# cause any problems.
# nltk.download('wordnet')

directory = './original_texts'

# All characters whose names appear at least 50 times in all three books.
# Extracted by review of output from methods below.
character_names = [
    'frodo',
    'sam',
    'gandalf',
    'aragorn',
    'pippin',
    'merry',
    'gollum',
    'gimli',
    'legolas',
    'faramir',
    'bilbo',
    'boromir',
    'saruman',
    'strider',
    'éomer',
    'théoden',
    'elrond',
    'sméagol',
    'treebeard',
    'denethor',
    'sauron',
    'tom',
    'éowyn',
    'galadriel',
    'lórien',
    'shadowfax',
    'beregond',
    'butterbur',
    'wormtongue',
    'isildur',
    'uglúk'
]


def get_processed_data(directory: str):
    file_names = [
        'the_fellowship_of_the_ring_trimmed.txt',
        'the_return_of_the_king_trimmed.txt',
        'the_two_towers_trimmed.txt'
    ]

    raw_text = ''

    for file_name in file_names:
        path = os.path.join(directory, file_name)
        with open(path, 'r') as file:
            raw_text += file.read()

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

    # Turns out that introduces some new multiple white spaces:
    text = ' '.join(text.split())

    # Remove numbers:
    numbers = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
    ]
    for number in numbers:
        text = text.replace(number, '')

    # Make it all lower case
    text = text.lower()
    text = text.split()

    print(f"Text Length = {len(text)}")

    vocabulary = list(set(text))

    print(f"Vocabulary Length = {len(vocabulary)}")

    word_counts = {
        word: text.count(word) for word in vocabulary
    }

    return text, vocabulary, word_counts


def print_words_with_counts(word_counts: dict[str, int],
                            reverse: bool = True):
    paired_data = sorted(
        word_counts.items(), reverse=reverse, key=lambda pair: pair[1]
    )
    for word, count in paired_data:
        print(f"{word}: {count}")


def get_adjectives(vocabulary: list[str]):
    adjectives = list()
    for word in vocabulary:
        synsets = wn.synsets(word)
        for synset in synsets:
            if synset._pos == 'a' and word not in adjectives:
                adjectives.append(word)
    return adjectives


def get_proximity_dataframe(text: list[str],
                            targets: list[str],
                            neighbors: list[str],
                            window: int = 10) -> pd.DataFrame:
    data = {
        target: {neighbor: 0 for neighbor in neighbors}
        for target in targets
    }
    divisors = {target: 0 for target in targets}
    for i, word in enumerate(text):
        # See if word is one of our targets:
        if word not in targets:
            continue
        divisors[word] += 1
        # Find the beginning and end of the window to search for neighbors:
        back_step = window
        forward_step = window
        while 'The Sun Still Burns':
            try:
                first_i = i - back_step
                test = text[first_i]
                break
            except IndexError:
                back_step += 1
        while 'The Earth Still Turns':
            try:
                last_i = i + forward_step
                test = text[last_i]
                break
            except IndexError:
                forward_step -= 1
        # Search for neighbors:
        for j in list(range(first_i, i)) + list(range(i, last_i + 1)):
            if text[j] in neighbors:
                data[word][text[j]] += \
                    ((window - (abs(i - j) - 1)) / window) ** 2
    data = np.array([list(x.values()) for x in data.values()])
    df = pd.DataFrame(data, index=targets, columns=neighbors)
    df = df.loc[:, (df != 0.0).any(axis=0)]
    s = df.sum()
    df = df[s.sort_values(ascending=False).index[:]]
    for target in targets:
        if target in df.index:
            df.loc[target] = df.loc[target]/divisors[target]
    return df


text, vocabulary, word_counts = get_processed_data(directory)
adjectives = get_adjectives(vocabulary)
# df = get_proximity_dataframe(text, character_names, adjectives)
# df.to_csv('lotr_adj_df_2021_11_03.csv')

paired_data = sorted(
    word_counts.items(), reverse=True, key=lambda pair: pair[1]
)
for word, count in paired_data:
    if word in adjectives:
        print(f"{word}: {count}")
