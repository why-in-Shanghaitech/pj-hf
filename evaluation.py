# evaluation.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Callable, List, Any
import argparse
import evaluate
import util


def get_macro_f1_metric() -> Callable[[List[Any], List[Any]], float]:
    """
    Question:
        Load the macro F1 metric from the Hugging Face Evaluation library.
    
    Returns:
        func (Callable): A function that takes two lists of integers as input and returns the macro F1 score.
    
    Example:
        >>> func = get_macro_f1_metric()
        >>> preds = [0, 2, 1, 0, 0, 2]
        >>> golds = [0, 1, 2, 0, 1, 2]
        >>> func(preds, golds)
        0.43333333333333335
    """

    """YOUR CODE HERE"""
    util.raiseNotDefined()


def main():
    parser = argparse.ArgumentParser(description="Metric Evaluation")
    parser.add_argument("--metric", type=str, help="Metric name", choices=["accuracy", "f1"], default="accuracy")

    args = parser.parse_args()

    if args.metric == "accuracy":
        # Example usage with accuracy metric
        preds = [0, 2, 1, 0, 0, 2]
        golds = [0, 1, 2, 0, 1, 2]

        accuracy_metric = evaluate.load("accuracy")
        accuracy = accuracy_metric.compute(predictions=preds, references=golds)
        print(accuracy) # {'accuracy': 0.5}

    elif args.metric == "f1":
        # Example usage with F1 metric
        preds = [0, 2, 1, 0, 0, 2]
        golds = [0, 1, 2, 0, 1, 2]

        func = get_macro_f1_metric()
        print(func(preds, golds)) # 0.43333333333333335


if __name__ == "__main__":
    main()
