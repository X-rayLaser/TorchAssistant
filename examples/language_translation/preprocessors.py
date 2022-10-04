"""
BSD 3-Clause License

Copyright (c) 2022, Evgenii Dolotov
Copyright (c) 2017-2022, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from torchassistant.preprocessors import ValuePreprocessor
from torchassistant.utils import Serializable


class SentenceEncoder(ValuePreprocessor, Serializable):
    """Based on Lang class from
     https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
     """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "OOV", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count OOV, SOS and EOS

    def fit(self, dataset):
        pass

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def process(self, value):
        sos = 1
        sentence = [self.word2index.get(word, 0) for word in value.split(' ')]
        eos = 2
        return [sos] + sentence + [eos]

    def __call__(self, value):
        return self.process(value)

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict.copy()
        self.index2word = {int(k): v for k, v in self.index2word.items()}


class FrenchEncoder(SentenceEncoder):
    def fit(self, dataset):
        for french_sentence, _ in dataset:
            self.add_sentence(french_sentence)

    @property
    def num_french_words(self):
        return self.n_words


class EnglishEncoder(SentenceEncoder):
    def fit(self, dataset):
        for _, english_sentence in dataset:
            self.add_sentence(english_sentence)

    @property
    def num_english_words(self):
        return self.n_words
