from __future__ import annotations

import numpy as np
import pandas as pd

import parameters as p
from tools.preprocessing import tag_w_pos_nlp, load_and_prep_files

TEXTS_DIR = p.TEXTS_DIR


def main():
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
    file_names = [
        'the_fellowship_of_the_ring_trimmed.txt',
        'the_return_of_the_king_trimmed.txt',
        'the_two_towers_trimmed.txt'
    ]
    text, vocabulary, word_counts = load_and_prep_files(
        file_names, TEXTS_DIR
    )
    tokenized_text = tag_w_pos_nlp(text)
    df = get_proximity_dataframe(tokenized_text, character_names)
    df.to_csv('../output/lotr_adj_df_nlp_2021_11_03.csv')
    pass


def get_proximity_dataframe(text: list[tuple[str, str]],
                            targets: list[str],
                            window: int = 10) -> pd.DataFrame:
    neighbors = [
        word[0] for word in text
        if word[1] in ['JJ', 'JJR', 'JJS'] and word[1] not in targets
    ]
    exclude = targets + ['i', 'mr', 'ive', 's', 'nay', 'o', 'im', 'cant'
                         'oh', 'sams', 'wont', 'yes', 'dun']
    neighbors = list(set(neighbors))
    data = {
        target: {neighbor: 0 for neighbor in neighbors}
        for target in targets
    }
    divisors = {target: 0 for target in targets}
    for i, word in enumerate(text):
        # See if word is one of our targets:
        if word[0] not in targets:
            continue
        divisors[word[0]] += 1
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
            if text[j][1] in ['JJ', 'JJR', 'JJS'] and text[j][0] not in exclude:
                data[word[0]][text[j][0]] += \
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


if __name__ == '__main__':
    main()