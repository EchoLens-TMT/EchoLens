import collections
import re

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.next_index = 4

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenizer(text):
        # very simple whitespace+punctuation splitter
        text = text.strip().lower()
        # split on any non-word character
        return [t for t in re.split(r"\W+", text) if t]

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for token, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.word2idx[token] = self.next_index
                self.idx2word[self.next_index] = token
                self.next_index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        numericalized = [self.word2idx["<start>"]]
        for token in tokenized_text:
            idx = self.word2idx.get(token, self.word2idx["<unk>"])
            numericalized.append(idx)
        numericalized.append(self.word2idx["<end>"])
        return numericalized