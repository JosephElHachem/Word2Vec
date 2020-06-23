import io
import itertools
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en.lex_attrs import _num_words

def word_count(sentences):
    counts = {word:0 for word in (itertools.chain.from_iterable(sentences))}
    for sentence in sentences:
        for word in sentence:
            counts[word] += 1
    return counts

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    # ( " -- . , ) ] [ ? ! : ; -
    ponctuations = ["'", '(', ')',']', '[', '"', "''", "`", "&", '--', '.', '?', '!', ':', ';', '-', '$', '}','{']
    # should we include stopwords? ex: the, that, which, for ...
    sentences = []
    with io.open(path, encoding='latin-1') as f:
        for l in f:
            # splitting sentences
            for pc in ['.', '?', '!']:
                l = l.replace(pc, '<END>')
            l = l.split('<END>')
            for sub_l in l:
                for pc in ponctuations:
                    sub_l = sub_l.replace(pc, ' ')
                # removing numbers
                sub_l = sub_l.lower().split()
                corrected_l = []
                previous_is_number = False
                for word in sub_l:
                    if word in ['nt', 't', 'not']:
                        corrected_l.append('<NEGATIVE>')
                        previous_is_number = False

                    elif word.isnumeric() or word in _num_words:
                        if previous_is_number:
                            continue
                        else:
                            corrected_l.append('<NUMBER>')
                            previous_is_number = True

                    elif word in STOP_WORDS:
                        continue

                    elif word.isalpha():
                        corrected_l.append(word)
                        previous_is_number = False
                if len(corrected_l)>=2:
                    sentences.append(corrected_l)
    return sentences

text2sentences('trainset.txt')


def loadPairs(path):
    '''
    For testing.
    '''
    data = pd.read_csv(path, delimiter='\t', header=0)
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs
