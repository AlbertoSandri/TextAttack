"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class GreedyWordSwapWIR(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(
        self,
        wir_method="unk",
        unk_token="[UNK]",
        wir_file_name=None,
        precomputed_idxs=None,
        logistic_regression=None,
        pca=None,
    ):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.index_order = None
        self.search_over = False
        self.wir_file_name = wir_file_name
        self.precomputed_idxs = precomputed_idxs
        self.logistic_regression = logistic_regression
        self.pca = pca
        self.texts_cache = {}

    def _get_index_order(self, initial_text, max_len=-1):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in indices_to_order:
                # Exit Loop when search_over is True - but we need to make sure delta_ps
                # is the same size as softmax_saliency_scores
                if search_over:
                    delta_ps = delta_ps + [0.0] * (
                        len(softmax_saliency_scores) - len(delta_ps)
                    )
                    break

                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, search_over = self.get_goal_results(
                    transformed_text_candidates
                )
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(len_text)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, index in enumerate(indices_to_order):
                matched_tokens = word2token_mapping[index]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]

        return index_order, search_over

    def perform_search(self, initial_result, restart=False):
        attacked_text = initial_result.attacked_text
        self.texts_cache = {}

        # Sort words by order of importance
        if not restart:
            if self.precomputed_idxs:
                try:
                    self.search_over = False
                    self.index_order = np.array(
                        self.precomputed_idxs[attacked_text.text]
                    )
                except KeyError:  # need this for sample 14 that has only 1 index
                    print(f"\n\n\nKeyError for: '{attacked_text.text}'\n\n\n")
                    self.index_order, self.search_over = self._get_index_order(
                        attacked_text
                    )
            else:
                self.index_order, self.search_over = self._get_index_order(
                    attacked_text
                )
        if self.wir_file_name:
            print(attacked_text)
            with open(self.wir_file_name, "a") as file:
                file.write(f"{self.index_order}\n")
                file.write(f"Text\n")
                words = attacked_text.words
                file.write(f"{words}\n")
                file.write(f"Indexes\n")
                indexes = [i for i, _ in enumerate(words)]
                file.write(f"{indexes}\n")

            return initial_result
        i = 0
        cur_result = initial_result
        results = None
        while i < len(self.index_order) and not self.search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[self.index_order[i]],
            )
            i += 1
            if self.logistic_regression:
                transformed_text_candidates = self.filter_candidates(
                    transformed_text_candidates, threshold=0.3386383103514465
                )
            if len(transformed_text_candidates) == 0:
                continue
            results, self.search_over = self.get_goal_results(
                transformed_text_candidates
            )
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]

    def filter_candidates(self, candidates, threshold=0.5, return_one=False):

        # Check there are candidates
        if len(candidates) == 0:
            return candidates

        # Cached texts
        filtered_candidates = [
            candidate
            for candidate in candidates
            if candidate.text in self.texts_cache.keys()
            and self.texts_cache[candidate.text] < threshold
        ]

        # No cached texts
        texts_no_cache = [
            candidate.text
            for candidate in candidates
            if candidate.text not in self.texts_cache.keys()
        ]

        # Get embeddings for candidates
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        embeddings_no_cache = compute_embeddings(texts_no_cache, model_name=model_name)
        embeddings = pd.DataFrame(embeddings_no_cache.numpy())
        # embeddings are pandas dataframe (n candidates, embedding size)

        # Reduce embeddings using PCA
        if self.pca:
            embeddings = self.pca.transform(embeddings)
        # obtain numpy array (n candidates, reduced embedding size (2))

        # Get predictions from logistic regression
        toxic_probs = self.logistic_regression.predict_proba(embeddings)[:, 1]

        # Filter candidates with toxic_probs > 0.5
        filtered_candidates.extend(
            [
                candidate
                for candidate, prob in zip(candidates, toxic_probs)
                if prob < threshold
            ]
        )

        # Update cache
        for text, prob in zip(texts_no_cache, toxic_probs):
            self.texts_cache[text] = prob

        if return_one:
            if len(filtered_candidates) > 0:
                filtered_candidates_sorted = sorted(
                    filtered_candidates,
                    key=lambda candidate: self.texts_cache[candidate.text],
                )
                return [filtered_candidates_sorted[0]]

        return filtered_candidates


import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import transformers

transformers.logging.set_verbosity_error()


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return {key: val.squeeze(0) for key, val in inputs.items()}


def compute_embeddings(
    texts: list[str], model_name: str, batch_size: int = 32
) -> torch.Tensor:
    """Compute sentence embeddings using a pre-trained transformer model.

    Args:
        texts (list[str]): List of texts to compute embeddings for.
        model_name (str): Name of the pre-trained model to use.
        batch_size (int): Batch size.

    Returns:
        torch.Tensor: Tensor with the sentence embeddings. Shape: (num_texts, embedding_size)
    """

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=True
    )

    # Load model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move inputs to the correct device
            inputs = {key: val.to(device) for key, val in batch.items()}

            # Extract embeddings
            outputs = model(**inputs)

            # Extract last hidden state
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]

            # Compute sentence embedding by averaging the token embeddings
            sentence_embeddings = last_hidden_state.mean(
                dim=1
            )  # Shape: (batch_size, sequence_length, hidden_size)

            all_embeddings.append(sentence_embeddings.cpu())

    # Concatenate all embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings
