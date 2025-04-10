from datasets import load_dataset
import pandas as pd

def load_and_sample_imdb(n=25000, seed=42):
    dataset = load_dataset("imdb")
    df = dataset["train"].to_pandas().sample(n=n, random_state=seed)
    df = df[["text", "label"]]
    return df

def load_test_set(n=12000, seed=123):
    dataset = load_dataset("imdb")
    df = dataset["test"].to_pandas().sample(n=n, random_state=seed)
    return df[["text", "label"]]
