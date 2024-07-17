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
    def build(
        model_wrapper,
        use_all_transformations=True,
        is_tokenizer_whitebox=False,
        allow_toggle=False,
        wir_file_name=None,
        precomputed_idxs=None,
    ):
        #
        # Swap characters out from words. Choose the best of four potential transformations.
        #
        transformation_white = None
        if is_tokenizer_whitebox:
            if use_all_transformations:
                # We propose four similar methods:
                transformation_white = CompositeTransformation(
                    [
                        # (1) Swap: Swap two adjacent letters in the word.
                        WordSwapNeighboringCharacterSwap(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                        ),
                        # (2) Substitution: Substitute a letter in the word with a random letter.
                        WordSwapRandomCharacterSubstitution(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                            max_candidates=50,
                        ),
                        # (3) Deletion: Delete a random letter from the word.
                        WordSwapRandomCharacterDeletion(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                        ),
                        # (4) Insertion: Insert a random letter in the word.
                        WordSwapRandomCharacterInsertion(
                            random_one=False,
                            is_tokenizer_whitebox=is_tokenizer_whitebox,
                            is_oov=model_wrapper.is_oov,
                            max_candidates=50,
                        ),
                    ]
                )
            else:
                # We use the Combined Score and the Substitution Transformer to generate
                # adversarial samples, with the maximum edit distance difference of 30
                # (ϵ = 30).
                transformation_white = WordSwapRandomCharacterSubstitution(
                    random_one=False,
                    is_tokenizer_whitebox=is_tokenizer_whitebox,
                    is_oov=model_wrapper.is_oov,
                    max_candidates=50,
                )
        if use_all_transformations:
            # We propose four similar methods:
            transformation_black = CompositeTransformation(
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
            transformation_black = WordSwapRandomCharacterSubstitution()
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
        search_method = GreedyWordSwapWIR(
            wir_file_name=wir_file_name,
            precomputed_idxs=precomputed_idxs,
        )

        return Attack(
            goal_function=goal_function,
            constraints=constraints,
            transformation=(
                transformation_white if is_tokenizer_whitebox else transformation_black
            ),
            search_method=search_method,
            is_tokenizer_whitebox=is_tokenizer_whitebox,
            allow_toggle=allow_toggle,
            transformation_black=transformation_black,
        )
