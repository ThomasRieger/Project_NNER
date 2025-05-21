from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import os
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load LST20 dataset
try:
    raw_dataset = load_dataset("lst20", split="train")
except Exception as e:
    logger.error(f"Failed to load LST20 dataset: {e}")
    raise

# 2. Parse raw lines to sentence-wise format
def parse_lines_to_sentences(dataset_split: List[Dict]) -> List[List[List[str]]]:
    sentences = []
    current_sentence = []
    
    for i, item in enumerate(dataset_split):
        line = item.get('text', '').strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            parts = line.split('\t')
            if len(parts) == 4:
                current_sentence.append(parts)
            else:
                logger.warning(f"Skipping malformed line at index {i}: {line}")
    if current_sentence:
        sentences.append(current_sentence)
    
    sentence_lengths = [len(s) for s in sentences]
    logger.info(f"Sentence count: {len(sentences)}, Max length: {max(sentence_lengths)}, Min length: {min(sentence_lengths)}")
    return sentences

train_sentences = parse_lines_to_sentences(raw_dataset)

# 3. Create label mappings
def extract_labels(sentences: List[List[List[str]]]) -> List[str]:
    label_set = set()
    for sentence in sentences:
        for word in sentence:
            if len(word) > 2:
                label_set.add(word[2])  # NER label
    labels = sorted(label_set)
    logger.info(f"Extracted {len(labels)} unique labels: {labels}")
    return labels

labels = extract_labels(train_sentences)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 4. Convert to HuggingFace dataset format
def convert_to_hf_format(sentences: List[List[List[str]]]) -> List[Dict]:
    hf_data = []
    for sentence in sentences:
        tokens = [word[0] for word in sentence]
        ner_tags = [label2id[word[2]] for word in sentence]
        hf_data.append({"tokens": tokens, "labels": ner_tags})
    return hf_data

hf_train = convert_to_hf_format(train_sentences[:18000])
train_dataset = Dataset.from_list(hf_train)

# 5. Tokenization + align labels
try:
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

def tokenize_and_align_labels(batch: Dict) -> Dict:
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        max_length=510,  # Reserve space for special tokens
        padding="max_length",
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True
    )

    all_labels = []
    for i in range(len(batch["tokens"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    labels.append(batch["labels"][i][word_idx])
                except IndexError:
                    logger.error(f"IndexError at sentence {i}, word_idx {word_idx}, labels: {batch['labels'][i]}")
                    raise
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        all_labels.append(labels)

    position_ids = tokenized_inputs.get("position_ids", None)
    if position_ids is not None:
        max_position_id = position_ids.max().item()
        logger.info(f"Max position_id: {max_position_id}")
        if max_position_id >= 512:
            logger.error(f"Position IDs exceed max length (512): {max_position_id}")
            raise ValueError("Position IDs out of range")

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

try:
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, batch_size=1000)
except Exception as e:
    logger.error(f"Tokenization failed: {e}")
    raise

# 6. Load model
try:
    model = AutoModelForTokenClassification.from_pretrained(
        "airesearch/wangchanberta-base-att-spm-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    logger.info(f"Model max_position_embeddings: {model.config.max_position_embeddings}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# 7. Define training args
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=False
)

# 8. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    processing_class=tokenizer  # Updated from tokenizer
)

try:
    trainer.train()
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# 9. Save model
save_path = "./ner_model_final"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

logger.info(f"Model and tokenizer saved to: {save_path}")