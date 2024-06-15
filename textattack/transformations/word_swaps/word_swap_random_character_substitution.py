"""
Word Swap by Random Character Substitution
------------------------------------------------
"""

import numpy as np

# from textattack.shared import utils
from .word_swap import WordSwap


class WordSwapRandomCharacterSubstitution(WordSwap):
    """Transforms an input by replacing one character in a word with a random
    new character.

    Args:
        random_one (bool): Whether to return a single word with a random
            character deleted. If not set, returns all possible options.
    >>> from textattack.transformations import WordSwapRandomCharacterSubstitution
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapRandomCharacterSubstitution()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(
        self,
        random_one=True,
        is_tokenizer_whitebox=False,
        is_oov=None,
        max_candidates=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.random_one = random_one
        self.is_tokenizer_whitebox = is_tokenizer_whitebox
        self.is_oov = is_oov
        self.max_candidates = max_candidates

    def _get_replacement_words(self, word):
        """Returns returns a list containing all possible words with 1 letter
        substituted for a random letter."""
        if len(word) <= 1:
            return []

        candidate_words = []

        if self.random_one:
            if self.is_tokenizer_whitebox:
                for _ in range(self.max_candidates):
                    i = np.random.randint(0, len(word))
                    candidate_word = (
                        word[:i] + self._get_random_letter() + word[i + 1 :]
                    )
                    if self.is_oov(candidate_word):
                        candidate_words.append(candidate_word)
                        break
            else:
                i = np.random.randint(0, len(word))
                candidate_word = word[:i] + self._get_random_letter() + word[i + 1 :]
                candidate_words.append(candidate_word)
        else:
            for i in range(len(word)):
                candidate_word = word[:i] + self._get_random_letter() + word[i + 1 :]
                candidate_words.append(candidate_word)
            if (
                self.is_tokenizer_whitebox and candidate_words
            ):  # TODO need to consider the max_candidates
                oov_words = []
                for candidate_word in candidate_words:
                    if self.is_oov(candidate_word):  # TODO could do this in a batch
                        oov_words.append(candidate_word)
                return oov_words

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["random_one"]
