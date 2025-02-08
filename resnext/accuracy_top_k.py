"""Top-K Accuracy metric."""

import datasets
from sklearn.metrics import top_k_accuracy_score

import evaluate


_DESCRIPTION = """
Top-K Accuracy is the proportion of correct predictions among the total number of cases processed, where a prediction is considered correct if the true label is among the top K predicted labels. It can be computed with:
Top-K Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `list` of `float`): Target scores. These can be either probability estimates or non-thresholded decision values (as returned by decision_function on some classifiers). The binary case expects scores with shape (n_samples,) while the multiclass case expects scores with shape (n_samples, n_classes). In the multiclass case, the order of the class scores must correspond to the order of labels, if provided, or else to the numerical or lexicographical order of the labels in y_true. If y_true does not contain all the labels, labels must be provided
    references (`list` of `int`): Ground truth labels.
    k (`int`): Number of top elements to consider for computing accuracy. Defaults to 1.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights. Defaults to None.
    labels: (`list` of `int`): Multiclass only. List of labels that index the classes in y_score. If None, the numerical or lexicographical order of the labels in y_true is used. If y_true does not contain all the labels, labels must be provided.

Returns:
    top_k_accuracy (`float`): Top-K accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`. A higher score means higher accuracy.

Examples:

    Example 1-A simple example
        >>> top_k_accuracy_metric = evaluate.load("top_k_accuracy")
        >>> results = top_k_accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[[0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.4, 0.6, 0.0], [0.7, 0.3, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]], k=2)
        >>> print(results)
        {'top_k_accuracy': 0.666}

    Example 2-The same as Example 1, except with `normalize` set to `False`.
        >>> top_k_accuracy_metric = evaluate.load("top_k_accuracy")
        >>> results = top_k_accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[[0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.4, 0.6, 0.0], [0.7, 0.3, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]], k=2, normalize=False)
        >>> print(results)
        {'top_k_accuracy': 4.0}

    Example 3-The same as Example 1, except with `sample_weight` set.
        >>> top_k_accuracy_metric = evaluate.load("top_k_accuracy")
        >>> results = top_k_accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[[0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.4, 0.6, 0.0], [0.7, 0.3, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]], k=2, sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
        >>> print(results)
        {'top_k_accuracy': 0.916}
"""


_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AccuracyTopK(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                # multilabel case isn't covered
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html"
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        k=1,
        normalize=True,
        sample_weight=None,
        labels=None,
    ):
        return {
            f"top_{k}_accuracy": float(
                top_k_accuracy_score(
                    references,
                    predictions,
                    k=k,
                    normalize=normalize,
                    sample_weight=sample_weight,
                    labels=labels,
                )
            )
        }
