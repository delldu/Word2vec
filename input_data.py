import numpy
from collections import deque
numpy.random.seed(12345)

import pdb

class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

        # pdb.set_trace()
        # (Pdb) pp self.word_count
        # 8934

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        # (Pdb) ratio.shape,ratio.sum()
        # ((8934,), 0.999999999999991)
        count = numpy.round(ratio * sample_table_size)
        # (Pdb) count
        # array([642837.,  15567.,  18796., ...,   3312.,   2328.,   2328.])
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

        # pdb.set_trace()
        # (Pdb) self.sample_table
        # array([   0,    0,    0, ..., 8933, 8933, 8933])
        # (Pdb) self.sample_table.shape
        # (100000559,)

    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())

        # pdb.set_trace()
        # (Pdb) len(batch_pairs), batch_pairs
        # (50, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 3),
        # (2, 4), (2, 5), (2, 5), (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 5), (3, 6), (4, 0), (4, 1),
        # (4, 2), (4, 3), (4, 5), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
        # (5, 6), (5, 7), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 5), (5, 7), (6, 2), (6, 3), (6, 4),
        # (6, 5), (6, 5)])

        return batch_pairs

    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()

        # pdb.set_trace()
        # (Pdb) a
        # self = <input_data.InputData object at 0x7f951c6aa0b8>
        # batch_size = 50
        # window_size = 5
        # (Pdb) type(neg_v), len(neg_v)
        # (<class 'list'>, 50)

        return neg_v

    def evaluate_pair_count(self, window_size):
        # pdb.set_trace()
        # (Pdb) self.sentence_length * (2 * window_size - 1) 
        # - (self.sentence_count - 1) * (1 + window_size) * window_size
        # 4264098
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
