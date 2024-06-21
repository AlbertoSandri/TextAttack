"""
Word Swap
-------------------------------
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``WordSwap`` class by overriding ``self._get_replacement_words``

"""

import random
import string

import torch

from textattack.transformations import Transformation
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder


class WordSwap(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing
    some of its words.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)
    """

    def __init__(
        self,
        letters_to_insert=None,
        **kwargs,
    ):
        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            self.letters_to_insert = string.ascii_letters
        self.is_tokenizer_whitebox = kwargs.get("is_tokenizer_whitebox", False)
        self.is_oov = kwargs.get("is_oov", None)
        self.use_scorer = kwargs.get("use_scorer", None)
        self.original_text = None

    def _get_replacement_words(self, word):
        """Returns a set of replacements given an input word. Must be overriden
        by specific word swap transformations.

        Args:
            word: The input word to find replacements for.
        """
        raise NotImplementedError()

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        # If this condition is met, it means that there is a new text
        if len(current_text.attack_attrs["modified_indices"]) == 0:
            self.original_text = current_text
        if self.is_tokenizer_whitebox and transformed_texts:
            if self.use_scorer:
                # Pick the best transformation according to USE
                transformed_texts = self.use_scorer.get_best_transformation(
                    self.original_text, transformed_texts
                )
            else:
                # Pick a random one
                transformed_texts = [random.choice(transformed_texts)]

        return transformed_texts
