# @Time    : 10/11/18 4:30 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : sentence.py

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class TF_Sentence(object):
    '''
    This is a tensorflow universal sentence encoder model
    https://tfhub.dev/google/universal-sentence-encoder-large/3
    '''

    def __init__(self, model_path):
        '''
        :param model_path: must be a directory which is ruled by tf-hub
        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # os.system('export TFHUB_CACHE_DIR=../nlp')
        # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        embed = hub.Module(model_path)
        self.input_placeholder = tf.placeholder(tf.string, shape=(None))
        self.output_placeholder = embed(self.input_placeholder)

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def encode(self, sentence):
        '''

        :param sentence: a word, a sentence or paragraph is string
        :return: a 512-D vector
        '''
        assert isinstance(sentence, str)
        sentence = [sentence]

        sentence_embedding_meta = self.session.run(
            self.output_placeholder, feed_dict={self.input_placeholder: sentence})

        vector = np.array(sentence_embedding_meta[0]).tolist()

        # print("Message: {}".format(sentence))
        # print("Embedding size: {}".format(len(vector)))
        # print("Embedding: {}".format(vector))
        return vector

    def run_and_plot(self, sentences):
        '''

        :param sentences: a list which contains many sentences
        :return: a similarity figure
        '''
        assert isinstance(sentences, list)
        message_embeddings_ = self.session.run(
            self.output_placeholder, feed_dict={self.input_placeholder: sentences})
        plot_similarity(sentences, message_embeddings_, 90)


def plot_similarity(labels, features, rotation):
    # plt.figure()
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    plt.show()
