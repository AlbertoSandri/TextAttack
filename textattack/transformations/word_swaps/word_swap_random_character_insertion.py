"""
Word Swap by Random Character Insertion
------------------------------------------------

"""

import numpy as np

# from textattack.shared import utils
from .word_swap import WordSwap


class WordSwapRandomCharacterInsertion(WordSwap):
    """Transforms an input by inserting a random character.

    random_one (bool): Whether to return a single word with a random
    character deleted. If not, returns all possible options.
    skip_first_char (bool): Whether to disregard inserting as the first
    character. skip_last_char (bool): Whether to disregard inserting as
    the last character.
    >>> from textattack.transformations import WordSwapRandomCharacterInsertion
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapRandomCharacterInsertion()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(
        self,
        random_one=True,
        skip_first_char=False,
        skip_last_char=False,
        is_tokenizer_whitebox=False,
        is_oov=None,
        max_candidates=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char
        self.is_tokenizer_whitebox = is_tokenizer_whitebox
        self.is_oov = is_oov
        self.max_candidates = max_candidates

    def _get_replacement_words(self, word):
        """Returns returns a list containing all possible words with 1 random
        character inserted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = (len(word) - 1) if self.skip_last_char else len(word)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            if self.is_tokenizer_whitebox:
                for _ in range(self.max_candidates):
                    i = np.random.randint(start_idx, end_idx)
                    candidate_word = word[:i] + self._get_random_letter() + word[i:]
                    if self.is_oov([candidate_word])[0]:
                        candidate_words.append(candidate_word)
                        break
            else:
                i = np.random.randint(start_idx, end_idx)
                candidate_word = word[:i] + self._get_random_letter() + word[i:]
                candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx):
                candidate_word = word[:i] + self._get_random_letter() + word[i:]
                candidate_words.append(candidate_word)
            if self.is_tokenizer_whitebox and candidate_words:
                is_oov_words = self.is_oov(candidate_words)
                candidate_words = [
                    candidate_words[i]
                    for i, is_oov in enumerate(is_oov_words)
                    if is_oov
                ]

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["random_one"]
