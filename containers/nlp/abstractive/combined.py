from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline, set_seed

import matplotlib.pyplot as plt
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, Dataset
import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch

from transformers import DataCollatorForSeq2Seq

from transformers import TrainingArguments, Trainer

from summarizer import Summarizer

nltk.download("punkt")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

model_ckpt = "google/pegasus-cnn_dailymail"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

abs_model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

ext_model = Summarizer(
        model="bert-large-uncased",
        hidden=-2,
        reduce_option='mean'
    )

def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]
        
        
def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                               batch_size=16, device=device,
                               column_text="article",
                               column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):

        inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                        padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device),
                         length_penalty=0.8, num_beams=8, max_length=128)
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

        # Finally, we decode the generated texts,
        # replace the  token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
               for s in summaries]

        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]


        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE scores.
    score = metric.compute()
    return score



#sample_text = cnn_dailymail["test"][0]["article"]
#reference = cnn_dailymail["test"][0]["highlights"]
#model(cnn_dailymail_train[i]["article"], num_sentences = 10)


cnn_dailymail_train = load_dataset('cnn_dailymail', '3.0.0', split = "train[:20%]")
cnn_dailymail_test = load_dataset('cnn_dailymail', '3.0.0', split = "test[:20%]")
cnn_dailymail_validation = load_dataset('cnn_dailymail', '3.0.0', split = "validation[:20%]")

articles = []
highlights = []
ids = []
for i in range(len(cnn_dailymail_train)):  
    articles.append(model(cnn_dailymail_train[i]["article"], num_sentences = 10))
    highlights.append(cnn_dailymail_train[i]["highlights"])
    ids.append(cnn_dailymail_train[i]["id"])
train_ext = {"article": articles, "highlights": highlights, "id": ids}

articles = []
highlights = []
ids = []
for i in range(len(cnn_dailymail_test)):  
    articles.append(model(cnn_dailymail_test[i]["article"], num_sentences = 10))
    highlights.append(cnn_dailymail_test[i]["highlights"])
    ids.append(cnn_dailymail_test[i]["id"])
test_ext = {"article": articles, "highlights": highlights, "id": ids}

articles = []
highlights = []
ids = []
for i in range(len(cnn_dailymail_validation)):  
    articles.append(model(cnn_dailymail_validation[i]["article"], num_sentences = 10))
    highlights.append(cnn_dailymail_validation[i]["highlights"])
    ids.append(cnn_dailymail_validation[i]["id"])
validation_ext = {"article": articles, "highlights": highlights, "id": ids}

cnn_dailymail = datasets.DatasetDict({"train":Dataset.from_dict(train_ext), "validation": Dataset.from_dict(validation_ext), "test":Dataset.from_dict(test_ext)})

split_lengths = [len(cnn_dailymail[split])for split in cnn_dailymail]

print(f"Split lengths: {split_lengths}")
print(f"Features: {cnn_dailymail['train'].column_names}")
print("\narticle:")

print(cnn_dailymail["test"][1]["article"])

print("\nhighlights:")

print(cnn_dailymail["test"][1]["highlights"])


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['article'] , max_length = 1024, truncation = True )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['highlights'], max_length = 128, truncation = True )

    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

cnn_dailymail_pt = cnn_dailymail.map(convert_examples_to_features, batched = True)


seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=abs_model)


trainer_args = TrainingArguments(
    output_dir='pegasus-cnn_dailymail', num_train_epochs=2, warmup_steps=500,
    per_device_train_batch_size=2, per_device_eval_batch_size=2,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16
)

trainer = Trainer(model=abs_model, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=cnn_dailymail_pt["train"],
                  eval_dataset=cnn_dailymail_pt["validation"])
                  
trainer.train()

rouge_metric = load_metric('rouge')
score = calculate_metric_on_test_ds(
    cnn_dailymail['test'], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'article', column_summary= 'highlights'
)

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )



df = pd.DataFrame(rouge_dict, index = [f'pegasus'] )



abs_model.save_pretrained("pegasus-cnn_dailymail-model")

tokenizer.save_pretrained("tokenizer")

cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0')

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

sample_text = cnn_dailymail["test"][0]["article"]

reference = cnn_dailymail["test"][0]["highlights"]

gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

pipe = pipeline("summarization", model="pegasus-cnn_dailymail-model",tokenizer=tokenizer)

print("Article:")
print(sample_text)


print("\nReference Summary:")
print(reference)


print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])

print(df.to_string())