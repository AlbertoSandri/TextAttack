"""

DeepWordBug
========================================
(Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)

"""

from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from .attack_recipe import AttackRecipe


class DeepWordBugGao2018(AttackRecipe):
    """Gao, Lanchantin, Soffa, Qi.

    Black-box Generation of Adversarial Text Sequences to Evade Deep
    Learning Classifiers.

    https://arxiv.org/abs/1801.04354
    """

    @staticmethod
    def build(model_wrapper, use_all_transformations=True, is_tokenizer_whitebox=False):
        #
        # Swap characters out from words. Choose the best of four potential transformations.
        #
        if is_tokenizer_whitebox:
            if use_all_transformations:
                use_scorer = UniversalSentenceEncoder(metric="angular")
                # We propose four similar methods:
                transformation = CompositeTransformation(
                    [
                        # (1) Swap: Swap two adjacent letters in the word.
                        WordSwapNeighboringCharacterSwap(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                            use_scorer=use_scorer,
                        ),
                        # (2) Substitution: Substitute a letter in the word with a random letter.
                        WordSwapRandomCharacterSubstitution(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                            max_candidates=50,
                            use_scorer=use_scorer,
                        ),
                        # (3) Deletion: Delete a random letter from the word.
                        WordSwapRandomCharacterDeletion(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                            use_scorer=use_scorer,
                        ),
                        # (4) Insertion: Insert a random letter in the word.
                        WordSwapRandomCharacterInsertion(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                            max_candidates=50,
                            use_scorer=use_scorer,
                        ),
                    ],
                    is_tokenizer_whitebox=is_tokenizer_whitebox,
                    use_scorer=use_scorer,
                )
            else:
                # We use the Combined Score and the Substitution Transformer to generate
                # adversarial samples, with the maximum edit distance difference of 30
                # (ϵ = 30).
                transformation = WordSwapRandomCharacterSubstitution(
                    random_one=False,
                    is_tokenizer_whitebox=is_tokenizer_whitebox,
                    is_oov=model_wrapper.is_oov,
                    max_candidates=50,
                )
        else:
            if use_all_transformations:
                # We propose four similar methods:
                transformation = CompositeTransformation(
                    [
                        # (1) Swap: Swap two adjacent letters in the word.
                        WordSwapNeighboringCharacterSwap(),
                        # (2) Substitution: Substitute a letter in the word with a random letter.
                        WordSwapRandomCharacterSubstitution(),
                        # (3) Deletion: Delete a random letter from the word.
                        WordSwapRandomCharacterDeletion(),
                        # (4) Insertion: Insert a random letter in the word.
                        WordSwapRandomCharacterInsertion(),
                    ]
                )
            else:
                # We use the Combined Score and the Substitution Transformer to generate
                # adversarial samples, with the maximum edit distance difference of 30
                # (ϵ = 30).
                transformation = WordSwapRandomCharacterSubstitution()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(30))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)
