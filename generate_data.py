import os

from tools.nonspecific_adjacency import get_proximity_dataframe
from tools.preprocessing import load_and_prep_files, tag_w_pos_nlp
import parameters as p
from resources import source_specific_data as data

TEXTS_DIR = p.TEXTS_DIR
OUTPUT_DIR = p.OUTPUT_DIR


def main():
    # generate_data(data.lord_of_the_rings, 'lotr_non_specific_adjacency')
    # generate_data(data.lolita, 'lolita_non_specific_adjacency')
    # generate_data(data.pride_and_prejudice, 'p_and_p_non_specific_adjacency')
    generate_data(data.the_sound_and_the_fury,
                  'the_sound_and_the_fury_non_specific_adjacency')
    generate_data(data.the_gormenghast_trilogy,
                  'the_gormenghast_trilogy_non_specific_adjacency')
    generate_data(data.the_portrait_of_a_lady,
                  'the_portrait_of_a_lady_non_specific_adjacency')
    generate_data(data.the_inferno,
                  'the_inferno_non_specific_adjacency')


def generate_data(dataset: str,
                  file_name: str):
    text, vocabulary, word_counts = load_and_prep_files(
        dataset['File Names'], TEXTS_DIR
    )
    tokenized_text = tag_w_pos_nlp(text)
    df = get_proximity_dataframe(tokenized_text,
                                 dataset['Character Names'])
    destination = os.path.join(OUTPUT_DIR, file_name + '.csv')
    df.to_csv(destination)


if __name__ == '__main__':
    main()
