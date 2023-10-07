import numpy as np
from collections import Counter, defaultdict
from typing import List, NoReturn, Optional
from torch.utils.data import Dataset

SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, error_check=False) -> NoReturn:
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None
        self.error_check = error_check

    def _from_list(self, tokens: Optional[List[str]] = None):
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self.stoi) == len(self.itos)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def add_tokens(self, tokens: List[str]) -> NoReturn:
        errors = ["cl-", "loc-", "poss-", "qu-"]
        tokens[len(self.specials):] = sorted(tokens[len(self.specials):])
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            is_error = False
            if t not in self.itos:
                if self.error_check:
                    for e in errors:
                        if e in t:
                            is_error = True
                            if not (t.replace(e, "") in self.stoi.keys()):
                                t = t.replace(e, "")
                                self.itos.append(t)
                                self.stoi[t] = new_index

                if not is_error:
                    self.itos.append(t)
                    self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)


class GlossVocabulary(Vocabulary):

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        file: Optional[str] = None,
        error_check: Optional[bool] = False
    ) -> NoReturn:
        super().__init__(error_check)
        self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 1
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)
        self.pad_token = PAD_TOKEN
        self.sil_token = SIL_TOKEN

        if tokens is not None:
            self._from_list(tokens)

        assert self.stoi[SIL_TOKEN] == 0

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        gloss_sequences = []
        for array in arrays:
            sequence = []
            for i in array:
                sequence.append(self.itos[i])
            gloss_sequences.append(sequence)
        return gloss_sequences


def filter_min(counter: Counter, minimum_freq: int):
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])  # sort by token (?)
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens


def build_vocab(
    dataset: Dataset, max_size: int, *, min_freq: int = 1, error_check: bool = False
) -> Vocabulary:

    tokens = []
    for glosses in dataset.glosses_list:
        tokens.extend(glosses)
    counter = Counter(tokens)
    if min_freq > -1:
        counter = filter_min(counter, min_freq)
    vocab_tokens = sort_and_cut(counter, max_size)
    assert len(vocab_tokens) <= max_size

    vocab = GlossVocabulary(tokens=vocab_tokens, error_check=error_check)

    assert len(vocab) <= max_size + len(vocab.specials)
    assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN

    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab
