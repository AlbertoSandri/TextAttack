"""
Composite Transformation
============================================
Multiple transformations can be used by providing a list of ``Transformation`` to ``CompositeTransformation``

"""

import random

from textattack.shared import utils
from textattack.transformations import Transformation


class CompositeTransformation(Transformation):
    """A transformation which applies each of a list of transformations,
    returning a set of all optoins.

    Args:
        transformations: The list of ``Transformation`` to apply.
    """

    def __init__(self, transformations, **kwargs):
        if not (
            isinstance(transformations, list) or isinstance(transformations, tuple)
        ):
            raise TypeError("transformations must be list or tuple")
        elif not len(transformations):
            raise ValueError("transformations cannot be empty")
        self.transformations = transformations
        self.is_tokenizer_whitebox = kwargs.get("is_tokenizer_whitebox", False)
        self.use_scorer = kwargs.get("use_scorer", None)
        self.original_text = None

    def _get_transformations(self, *_):
        """Placeholder method that would throw an error if a user tried to
        treat the CompositeTransformation as a 'normal' transformation."""
        raise RuntimeError(
            "CompositeTransformation does not support _get_transformations()."
        )

    def __call__(self, *args, **kwargs):
        new_attacked_texts = set()
        for transformation in self.transformations:
            new_attacked_texts.update(transformation(*args, **kwargs))

        transformed_texts = list(new_attacked_texts)

        return_indices = kwargs.get("return_indices", False)

        if return_indices:
            # Save the original text
            self.original_text = args[0]
        else:
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

    def __repr__(self):
        main_str = "CompositeTransformation" + "("
        transformation_lines = []
        for i, transformation in enumerate(self.transformations):
            transformation_lines.append(utils.add_indent(f"({i}): {transformation}", 2))
        transformation_lines.append(")")
        main_str += utils.add_indent("\n" + "\n".join(transformation_lines), 2)
        return main_str

    __str__ = __repr__
