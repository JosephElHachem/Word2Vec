import os, time
import argparse
import numpy as np
import pickle as pkl
from scipy.special import expit
from utils import *

__author__ = ['joseph El Hachem']
__email__  = ['joseph.el-hachem@student.ecp.fr']


class SkipGram:
    def __init__(self, sentences, stopWords=STOP_WORDS, n_epochs=20, start_from=None, save_path=None, checkpoint_save=1, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5, learning_rate=1e-2, checkpoint=1000, table_length=1e8):
        self.save_path = save_path
        self.test_set_path = 'train/ground_truth.txt'
        self.ground_truth = None
        self.n_epochs = n_epochs
        self.checkpoint = checkpoint
        self.checkpoint_save = checkpoint_save
        self.current_trainset = sentences # set of sentences
        self.stopWords = stopWords
        # saving hyperparams of the model
        self.winSize = winSize
        self.minCount = minCount
        self.negativeRate = negativeRate
        # remove unfrequent words in self.current_trainset (occurences<minCount)
        self.remove_infrequent()
        self.nEmbed = nEmbed

        if start_from is None:
            self._init_start_from_none(sentences)
        else:
            self._init_start_from_previous(sentences, start_from)

        self.frequencies = self.get_frequencies() # should the frequencies change ?
        self.unigram_table = self.compute_unigrams(table_length=table_length)
        # loss
        self.trainWords = 1
        self.accLoss = 0
        self.lr = learning_rate
        pass

    def _init_start_from_previous(self, sentences, start_from):
        previous_vocab = self.load(start_from, to_load='vocab')
        previous_length = int(len(previous_vocab))
        previous_w2id = self.load(start_from, to_load='w2id')
        previous_id2w = self.load(start_from, to_load='id2w')
        previous_hidden_layer = self.load(start_from, to_load='hidden_layer')
        previous_output_layer = self.load(start_from, to_load='output_layer')
        self.nEmbed = previous_hidden_layer.shape[1]
        self.loss = self.load(start_from, to_load='loss')
        # current vocab is the previous vocab union to the new vocab
        self.vocab = set(itertools.chain.from_iterable(sentences)).union(previous_vocab)  # union of sets
        self.vocab_length = len(self.vocab)
        ## new words to expand our w2id, id2w and network layers
        new_words = self.vocab.difference(previous_vocab)
        nb_new_words = len(new_words)
        # w2id. Mapping with integers, starting from the index we stopped at last time (=previous_length)
        new_w2id = {word: i + previous_length for i, word in enumerate(new_words)}  # word to Id mapping
        self.w2id = dict(new_w2id, **previous_w2id)
        # id2w
        new_id2w = {i + previous_length: word for i, word in enumerate(new_words)}  # Id to word mapping
        new_id2w.update(previous_id2w)
        self.id2w = new_id2w
        # concatenating new lines at the bottom of hidden_layer for the new words
        new_hidden_layer = np.random.uniform(-1, 1, (nb_new_words, self.nEmbed))
        new_output_layer = np.random.uniform(-1, 1, (self.nEmbed, nb_new_words))
        self.hidden_layer = np.vstack((previous_hidden_layer, new_hidden_layer))
        self.output_layer = np.hstack((previous_output_layer, new_output_layer))
        del previous_vocab
        del previous_length
        del previous_w2id
        del previous_id2w
        del previous_hidden_layer
        del previous_output_layer

    def _init_start_from_none(self, sentences):
        self.vocab = set(itertools.chain.from_iterable(sentences))
        self.vocab_length = int(len(self.vocab))
        self.w2id = {word: i for i, word in enumerate(self.vocab)}  # word to Id mapping
        self.id2w = {i: word for i, word in enumerate(self.vocab)}  # Id to word mapping
        # layers
        self.hidden_layer = np.random.uniform(-1, 1, (self.vocab_length, self.nEmbed))  # vocab_length x 100
        self.output_layer = np.random.uniform(-1, 1, (self.nEmbed, self.vocab_length))  # 100 x vocab_length
        self.loss = []

    def remove_infrequent(self):
        counts = word_count(self.current_trainset)
        to_remove = []
        for i, sentence in enumerate(self.current_trainset):
            replacement = []
            for word in sentence:
                if counts[word] >= self.minCount:
                    replacement.append(word)
            self.current_trainset[i] = replacement

            # after removing unfrequent words and stop words, check sentence length.
            if len(replacement) < 2: # sentence to short to train on (single word: example: Don't! ).
                to_remove.append(i)
        self.index_deleted = []
        for index in sorted(to_remove, reverse=True):
            self.index_deleted.append(index)
            del self.current_trainset[index]
        self.nb_words = sum(list(map(len, self.current_trainset)))
        pass

    def get_frequencies(self):
        '''
        calculates frequencies of words in self.current_trainset with power of 0.75 as in the paper.
        return: list of frequencies, ordered as w2id
        '''
        frequencies = np.array([0 for i in range(self.vocab_length)])
        # word count
        for sentence in self.current_trainset:
            for word in sentence:
                frequencies[self.w2id[word]] += 1
        frequencies = frequencies**0.75
        frequencies = frequencies/np.sum(frequencies)
        return frequencies

    def compute_unigrams(self, table_length=1e8):
        '''
        Generates table containing ids of all vocab words with frequencies obtained from get_frequencies: f[word]**3/4
        '''
        table = np.zeros(int(table_length))
        if not hasattr(self, 'frequencies'):
            self.get_frequencies()

        counter = 0
        idx = 0
        while counter < table_length and idx < self.vocab_length:
            size = int(self.frequencies[idx] * table_length)
            table[counter:counter+size] = idx # if counter+size >= table_length, it will stop at table_length-1
            counter += size
            idx += 1
        if 0 < counter < table_length: # quit because of vocab is finished
            table[counter:] = table[np.random.randint(0, counter)]
        if 0 < idx < self.vocab_length: # quit because no more place in table
            raise ValueError('not all words are represented in unigram computation. Increase table_length')
        np.random.shuffle(table)
        return table.astype(int)

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        neg_inds = []
        while len(neg_inds) < self.negativeRate:
            raw_samp = np.random.choice(self.unigram_table)
            if not raw_samp in omit:
                neg_inds.append(raw_samp)
        return np.array(neg_inds)

    def train(self):
        t0 = time.time()
        for epoch_id in range(1, self.n_epochs+1):
            print(f'Starting with epoch {epoch_id}')
            for counter, sentence in enumerate(self.current_trainset):
                # Looping on the sentence to train for each word
                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)  # index of the start context
                    end = min(wpos + winsize + 1, len(sentence)) # index of the end of the context
                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word] # id of the word used
                        if ctxtId == wIdx: continue # we're on the current word: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds)
                        self.trainWords += 1

                if counter>0 and counter % self.checkpoint == 0:
                    t1 = time.time()
                    hours, rem = divmod(t1 - t0, 3600)
                    minutes, seconds = divmod(rem, 60)
                    current_time = "{:0>2d}:{:0>2d}:{:0>2d}".format(int(hours), int(minutes), int(seconds))

                    self.loss.append(self.accLoss / self.trainWords)
                    self.trainWords = 0
                    self.accLoss = 0
                    print(f' > training sentence {counter} of {len(self.current_trainset)} ----- time = {current_time} ----- loss: {np.round(self.loss[-1],3)}')

            if epoch_id>0 and epoch_id % self.checkpoint_save == 0:
                if self.save_path is not None:
                    checkpoint_path = os.path.join(self.save_path, 'checkpoint', str(epoch_id))
                    if not os.path.isdir(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    self.save(checkpoint_path)
        if self.save_path is not None:
            self.save(self.save_path)

    def trainWord_old(self, wordId, contextId, negativeIds):
        # forward pass: NO NEED TO DO THE FORWARD PASS ON ALL OUTPUT
        idx = [contextId]+list(negativeIds)
        output = expit(np.dot(self.hidden_layer[wordId, :], self.output_layer[:, idx])) # shape: len(idx), 1
        # loss: difference between output probability and expected probability
        self.accLoss += 1 - output[0] + np.sum(output[1:])
        sig_1_sig = output * (1 - output)
        # sign for gradient step
        signs = np.array([1]+[-1 for i in negativeIds])
        # updates
        self.output_layer[:, idx]    += self.lr * signs * np.dot(self.hidden_layer[wordId, :].reshape(-1,1), sig_1_sig.reshape(1,-1))
        self.hidden_layer[wordId, :] += self.lr * (
            sig_1_sig[0] * self.output_layer[:, contextId] - np.dot(sig_1_sig[1:] * self.output_layer[:,negativeIds], np.ones(self.negativeRate))
        )

    def trainWord(self, wordId, contextId, negativeIds):
        '''
        trainWord with loss function as logsoftmax instead of softmax
        :param wordId: id of word as input
        :param contextId: id of context word to predict
        :param negativeIds: negative ids
        :return:
        '''
        # forward pass: NO NEED TO DO THE FORWARD PASS ON ALL OUTPUT
        idx = [contextId]+list(negativeIds)

        # word vecs
        word_vec = self.hidden_layer[wordId,:]
        ctxt_vec = self.output_layer[:,contextId]
        neg_vecs = self.output_layer[:,negativeIds]

        # softmax for gradients. reminder: softmax(-x) = 1 - softmax(x) and ∂softmax(x)/∂x = softmax(x) ( 1 - softmax(x))
        softmax_c = expit( np.dot(word_vec, ctxt_vec) )
        softmax_neg = expit( np.dot(word_vec, neg_vecs) )

        grad_word_vec = - (1-softmax_c) * ctxt_vec + np.dot(neg_vecs, softmax_neg.reshape(-1,1)).flatten()
        grad_ctxt_vec = - (1-softmax_c) * word_vec
        grad_neg_vecs = np.outer(word_vec, softmax_neg)

        # gradient update
        word_vec = word_vec - self.lr * grad_word_vec
        ctxt_vec = ctxt_vec - self.lr * grad_ctxt_vec
        neg_vecs = neg_vecs - self.lr * grad_neg_vecs

        # saving updates
        self.hidden_layer[wordId,:]  = word_vec
        self.output_layer[:,contextId] = ctxt_vec
        self.output_layer[:,negativeIds] = neg_vecs

        # loss update
        loss = - np.log(softmax_c) - np.sum(np.log(1 - softmax_neg))
        self.accLoss += loss

    def save(self,path):
        '''
        path: directory to save following: vocab, w2id, id2w, hidden_layer, output_layer
        '''
        model_info = {
            "nEmbed": self.nEmbed,
            "negativeRate": self.negativeRate,
            "winSize": self.winSize,
            "minCount": self.minCount,
            "learningRate":self.lr
        }

        if not os.path.isdir(path):
            os.mkdir(path)

        hidden_path = os.path.join(path, 'hidden_layer.npy')
        np.save(hidden_path, self.hidden_layer)

        output_path = os.path.join(path, 'output_layer.npy')
        np.save(output_path, self.output_layer)

        model_info_path = os.path.join(path, 'model_info.pkl')
        with open(model_info_path, 'wb') as F:
            pkl.dump(model_info, F)

        vocab_path = os.path.join(path, 'vocab.pkl')
        with open(vocab_path, 'wb') as F:
            pkl.dump(self.vocab, F)

        w2id_path = os.path.join(path, 'w2id.pkl')
        with open(w2id_path, 'wb') as F:
            pkl.dump(self.w2id, F)

        id2w_path = os.path.join(path, 'id2w.pkl')
        with open(id2w_path, 'wb') as F:
            pkl.dump(self.id2w, F)

        loss_path = os.path.join(path, 'loss.pkl')
        with open(loss_path, 'wb') as F:
            pkl.dump(self.loss, F)

        print(f'SAVING vocab.pkl, w2id.pkl, id2w.pkl, loss.pkl, hidden_layer.npy, output_layer.npy inside {path}')

    def encode(self,word):
        '''
        from word to it's embedding
        '''
        return self.hidden_layer[self.w2id[word], :]

    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float in [0,1] indicating the similarity (the higher the more similar)
        """
        if word1 in self.vocab and word2 in self.vocab:
            vec1 = self.encode(word1)
            vec2 = self.encode(word2)
            similarity_ = (1 + self.cosine_similarity(vec1, vec2))/2
        else:
            similarity_ = 0.5 + np.random.randn() * 0.1

        return similarity_

    def cosine_similarity(self, vec1, vec2):
        '''
        compute cosine similarity between two vectors.
        '''
        return np.dot(vec1,vec2)/ (np.linalg.norm(vec1)*np.linalg.norm(vec2))


    def most_similar(self, word, k=5):
        '''
        Returns the k most similar words in the vocab to the given word.
        '''
        scores = np.array([self.similarity(word, word_candidate) for word_candidate in list(self.vocab)])
        k_scores = np.argsort(scores)[-(k+1):-1] # best k scores, without the current word which always has best score
        words = [list(self.vocab)[idx] for idx in k_scores]
        return words, scores[k_scores]

    def load_model(self, path):
        '''
        load fully a model saved in path
        '''
        self.loss = self.load(path, to_load='loss')
        self.hidden_layer = self.load(path, to_load='hidden_layer')
        self.output_layer = self.load(path, to_load='output_layer')
        self.vocab = self.load(path, to_load='vocab')
        self.w2id = self.load(path, to_load='w2id')
        self.id2w = self.load(path, to_load='id2w')

    def test_model(self):
        if self.ground_truth is None:
            test_pairs = loadPairs(self.test_set_path)
            self.ground_truth = []
            for word1, word2, truth in list(test_pairs):
                self.ground_truth.append(truth)

        test_pairs = loadPairs(self.test_set_path)
        my_similarities = []
        for word1, word2, truth in list(test_pairs):
            my_similarities.append(self.similarity(word1, word2))
        correlation = np.corrcoef(self.ground_truth, my_similarities)[0][1]
        print(f'correlation to ground truth: {correlation}')

    @staticmethod
    def load(path, to_load=None):
        '''
        path: path to directory where items are stored
        to_load: string, indicates item to load. Possible values for to_load:
        vocab, w2id, id2w, hidden_layer, output_layer
        '''
        if to_load is None:
            text = "Precise item to load in 'to_load' as string.\nChoose from: vocab, w2id, id2w, hidden_layer, output_layer"
            raise ValueError(text)

        full_path = os.path.join(path, to_load)
        if to_load == 'hidden_layer' or to_load == 'output_layer':
            full_path += '.npy'
            out = np.load(full_path)
        else:
            full_path += '.pkl'
            with open(full_path, 'rb') as handle:
                out = pkl.load(handle)
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--n_epochs', help='number of epochs on training data', default=5, type=int)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences,
                      save_path=None,
                      minCount=5,
                      negativeRate=5,
                      stopWords=STOP_WORDS,
                      learning_rate=1e-2,
                      checkpoint=1000,
                      n_epochs=opts.n_epochs,
                      checkpoint_save=1)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram([])
        sg.load_model(opts.model)
        similarities = []
        for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            similarities.append(sg.similarity(a,b))
        print(f'mean  : {np.mean(similarities)}')
        print(f'std   : {np.std(similarities)}')
        print(f'median: {np.median(similarities)}')
