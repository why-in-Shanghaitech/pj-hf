# classification.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Callable
import argparse
from transformers import pipeline
import gradio as gr


def get_intent_classification_pipeline() -> Callable[[str], dict]:
    """
    Question:
        Load the pipeline for intent text classification.
        There are 5 possible labels: 
            'FACT', 'QUESTION', 'REQUEST', 'OPINION', 'COMPLAINT'
        Find a proper model from HuggingFace Model Hub, then load the pipeline to classify the text.
        Notice that we have time limits so you should not use a model that is too large. A model with 
        100M params is enough.

    Returns:
        func (Callable): A function that takes a string as input and returns a dictionary with the
        predicted label and its score.

    Example:
        >>> func = get_intent_classification_pipeline()
        >>> result = func("Can you send me the document?")
        {"label": "REQUEST", "score": 0.9999999403953552}
    """
    pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
    def func(text: str) -> dict:
        return pipe(text)[0]
    return func


def main():
    parser = argparse.ArgumentParser(description="Intent Classification Pipeline")
    parser.add_argument("--task", type=str, help="Task name", choices=["sentiment", "intent"], default="sentiment")
    parser.add_argument("--use-gradio", action="store_true", help="Use Gradio for UI")

    args = parser.parse_args()

    if args.use_gradio and args.task == "sentiment":
        # Example usage with Gradio
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        iface = gr.Interface.from_pipeline(pipe)
        iface.launch()

    elif args.use_gradio and args.task == "intent":
        # Visualize the intent classification pipeline with Gradio
        pipe = get_intent_classification_pipeline()
        iface = gr.Interface(
            fn=lambda x: {item["label"]: item["score"] for item in [pipe(x)]},
            inputs=gr.components.Textbox(label="Input", render=False),
            outputs=gr.components.Label(label="Classification", render=False),
            title="Text Classification",
        )
        iface.launch()

    elif not args.use_gradio and args.task == "sentiment":
        # Example usage
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        print(pipe("This movie is great!")[0]) # {'label': 'positive', 'score': 0.988831102848053}

    elif not args.use_gradio and args.task == "intent":
        # Test the function
        func = get_intent_classification_pipeline()
        print(func("Can you send me the document?")) # {"label": "REQUEST", "score": ...}


if __name__ == "__main__":
    main()
