"""

TextBugger
===============

(TextBugger: Generating Adversarial Text Against Real-world Applications)

"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from .attack_recipe import AttackRecipe


class TextBuggerLi2018(AttackRecipe):
    """Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).

    TextBugger: Generating Adversarial Text Against Real-world Applications.

    https://arxiv.org/abs/1812.05271
    """

    @staticmethod
    def build(
        model_wrapper,
        is_tokenizer_whitebox=False,
        is_bert_tokenizer_whitebox=False,
        allow_toggle=False,
        wir_file_name=None,
        precomputed_idxs=None,
        logistic_regression=None,
        pca=None,
        number_queries_file_name=None,
        number_words_file_name=None,
    ):
        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        transformation_white = None
        if is_bert_tokenizer_whitebox:
            transformation_white = WordSwapHomoglyphSwap(
                random_one=False,
                is_tokenizer_whitebox=is_bert_tokenizer_whitebox,
                is_oov=model_wrapper.is_oov,
            )
        elif is_tokenizer_whitebox:
            transformation_white = CompositeTransformation(
                [
                    # (1) Insert: Insert a space into the word.
                    # Generally, words are segmented by spaces in English. Therefore,
                    # we can deceive classifiers by inserting spaces into words.
                    WordSwapRandomCharacterInsertion(
                        random_one=False,
                        letters_to_insert=" ",
                        skip_first_char=True,
                        skip_last_char=True,
                        is_tokenizer_whitebox=is_tokenizer_whitebox,
                        is_oov=model_wrapper.is_oov,
                        max_candidates=50,
                    ),
                    # (2) Delete: Delete a random character of the word except for the first
                    # and the last character.
                    WordSwapRandomCharacterDeletion(
                        random_one=False,
                        skip_first_char=True,
                        skip_last_char=True,
                        is_tokenizer_whitebox=is_tokenizer_whitebox,
                        is_oov=model_wrapper.is_oov,
                    ),
                    # (3) Swap: Swap random two adjacent letters in the word but do not
                    # alter the first or last letter. This is a common occurrence when
                    # typing quickly and is easy to implement.
                    WordSwapNeighboringCharacterSwap(
                        random_one=False,
                        skip_first_char=True,
                        skip_last_char=True,
                        is_tokenizer_whitebox=is_tokenizer_whitebox,
                        is_oov=model_wrapper.is_oov,
                    ),
                    # (4) Substitute-C (Sub-C): Replace characters with visually similar
                    # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                    # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                    WordSwapHomoglyphSwap(
                        random_one=False,
                        is_tokenizer_whitebox=is_tokenizer_whitebox,
                        is_oov=model_wrapper.is_oov,
                    ),
                    # (5) Substitute-W
                    # (Sub-W): Replace a word with its topk nearest neighbors in a
                    # context-aware word vector space. Specifically, we use the pre-trained
                    # GloVe model [30] provided by Stanford for word embedding and set
                    # topk = 5 in the experiment.
                    WordSwapEmbedding(
                        max_candidates=5,
                        is_tokenizer_whitebox=is_tokenizer_whitebox,
                        is_oov=model_wrapper.is_oov,
                    ),
                ]
            )

        transformation_black = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        # In our experiment, we first use the Universal Sentence
        # Encoder [7], a model trained on a number of natural language
        # prediction tasks that require modeling the meaning of word
        # sequences, to encode sentences into high dimensional vectors.
        # Then, we use the cosine similarity to measure the semantic
        # similarity between original texts and adversarial texts.
        # ... "Furthermore, the semantic similarity threshold \eps is set
        # as 0.8 to guarantee a good trade-off between quality and
        # strength of the generated adversarial text."
        constraints.append(UniversalSentenceEncoder(threshold=0.8))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(
            wir_method="delete",
            wir_file_name=wir_file_name,
            precomputed_idxs=precomputed_idxs,
            logistic_regression=logistic_regression,
            pca=pca,
            number_queries_file_name=number_queries_file_name,
            number_words_file_name=number_words_file_name,
        )

        return Attack(
            goal_function=goal_function,
            constraints=constraints,
            transformation=(
                transformation_white
                if is_tokenizer_whitebox or is_bert_tokenizer_whitebox
                else transformation_black
            ),
            search_method=search_method,
            is_tokenizer_whitebox=is_tokenizer_whitebox or is_bert_tokenizer_whitebox,
            allow_toggle=allow_toggle,
            transformation_black=transformation_black,
        )
