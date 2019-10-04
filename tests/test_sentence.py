# @Time    : 10/11/18 6:22 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_sentence.py

import _init_paths
from models.nlp.sentence import TF_Sentence


if __name__ == '__main__':
    word = "Elephant"
    sentence = "I am a sentence for which I would like to get its embedding."
    paragraph = (
        "Universal Sentence Encoder embeddings also support short paragraphs. "
        "There is no hard limit on how long the paragraph is. Roughly, the longer "
        "the more 'diluted' the embedding will be.")
    messages = [word, sentence, paragraph]


    sentences = [
        # Smartphones
        "smartphone",
        "I bought a new phone yesterday.",

        # Clothes
        "Nike produces a new sports shirt.",
        "Your new sports shirt looks great.",

        # Food and health
        "Strawberry ice-cream",
        "Eating strawberries is healthy"
    ]

    m = TF_Sentence('../weights/sentence/96e8f1d3d4d90ce86b2db128249eb8143a91db73')

    for i in messages:
        m.encode(i)

    m.run_and_plot(sentences)