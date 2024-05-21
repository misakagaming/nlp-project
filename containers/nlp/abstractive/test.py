from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline, set_seed

import matplotlib.pyplot as plt
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, Dataset
import nltk
from nltk.tokenize import sent_tokenize
import sys
from tqdm import tqdm
import torch

from transformers import DataCollatorForSeq2Seq

from transformers import TrainingArguments, Trainer

from summarizer import Summarizer


ext_model = Summarizer(
        model="bert-large-uncased",
        hidden=-2,
        reduce_option='mean'
    )

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0', split = "test[:1%]")


sample_text = ext_model(cnn_dailymail[0]["article"], num_sentences=5)

reference = cnn_dailymail["test"][0]["highlights"]

gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

pipe = pipeline("summarization", model=".\pegasus-cnn_dailymail-model",tokenizer=tokenizer)

print("Article:")
print(sample_text)


print("\nReference Summary:")
print(reference)


print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])