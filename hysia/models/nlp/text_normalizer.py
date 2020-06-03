import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from zhzlib.preprocessing.contractions import CONTRACTION_MAP
import unicodedata


def sentence_filter(s):
    s_arr = s.split()
    for i, v in enumerate(s_arr):
        s_arr[i] = v.strip()
    temp = " ".join(s_arr)
    pattern = re.compile("[^\w +]+")
    result = re.sub(pattern, '', temp)
    return result


class TextNormalization():
    '''
    Clean the noisy text data
    1. remove html tags
    2. accented_chars
    3. expand contractions
    4. remove special characters
    5. text lemmatization
    6. text stemming
    7. remove stopwords
    '''

    def __init__(self):
        self.nlp = spacy.load('en')
        self.tokenizer = ToktokTokenizer()

        try:
            self.stopword_list = nltk.corpus.stopwords.words('english')
        except:
            nltk.download('stopwords')
            self.stopword_list = nltk.corpus.stopwords.words('english')
        self.stopword_list.remove('no')
        self.stopword_list.remove('not')

    # remove html tags
    @staticmethod
    def __strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    # remove accented characters
    @staticmethod
    def __remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    # expand contractions
    @staticmethod
    def __expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    # remove special characters
    @staticmethod
    def __remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    # text lemmatization
    # TODO this will lower case
    def __lemmatize_text(self, text):
        text = self.nlp(text)
        # for word in text:
        #     print(word)
        #     print(word.text)

        # lemma_: Base form of the token, with no inflectional suffixes.
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        # text = ' '.join([word.text if word.text != '-PRON-' else word.text for word in text])
        return text

    # text stemming
    @staticmethod
    def __simple_stemmer(text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    def __remove_stopwords(self, text, is_lower_case=False):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def normalize_corpus(self, corpus, html_stripping=False, contraction_expansion=True,
                             accented_char_removal=True, text_lower_case=True,
                         text_lemmatization=True, special_char_removal=True,
                         stopword_removal=False, remove_digits=False):

        doc = corpus

        # normalize each document in the corpus

        # strip HTML
        if html_stripping:
            doc = self.__strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = self.__remove_accented_chars(doc)
        # expand contractions
        if contraction_expansion:
            doc = self.__expand_contractions(doc)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        # lemmatize text
        if text_lemmatization:
            doc = self.__lemmatize_text(doc)
        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = self.__remove_special_characters(doc, remove_digits=remove_digits)
            # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = self.__remove_stopwords(doc, is_lower_case=text_lower_case)

        return doc


if __name__ == '__main__':
    full_text = 'life was like a box of chocolates, you never know what you’re gonna get'
    clean = TextNormalization()
    clean_text = clean.normalize_corpus(full_text)

    print(clean_text)
    print('\n')
    print(full_text)