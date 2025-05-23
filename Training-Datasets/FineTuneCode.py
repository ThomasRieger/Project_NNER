from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import os
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE = "Training-Datasets/Raw/_txt0001.txt"
FIXED_FILE = "Training-Datasets/Fix/_txt0001.txt"

# 0. Augment Data
def augment_data(input_file: str, output_file: str):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Original data
        augmented_lines = lines[:]
        # Add synthetic sentences with กะโนยโคยโทง
        synthetic_sentences = [
            "กะโนยโคยโทง\tNCMN\tB_PER\nเป็น\tVSTA\tO\nผู้นำ\tNCMN\tO\nชุมชน\tNCMN\tO\n\n",
            "นาย\tNCMN\tB_TTL\nกะโนยโคยโทง\tNCMN\tB_PER\nจาก\tPREP\tO\nจังหวัด\tNCMN\tB_LOC\nเชียงใหม่\tNCMN\tI_LOC\n\n",
            "กะโนยโคยโทง\tNCMN\tB_PER\nและ\tCONJ\tO\nทีม\tNCMN\tO\nทำงาน\tVACT\tO\nที่\tPREP\tO\nกรุงเทพ\tNCMN\tB_LOC\n\n"
        ]
        augmented_lines.extend(synthetic_sentences)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(augmented_lines))
        logger.info(f"Augmented file saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to augment data: {e}")
        raise

# Run data augmentation
augment_data(FILE, FILE + ".augmented")
FILE = FILE + ".augmented"

# 1. Fix File Format
def fix_file_format(input_file: str, output_file: str):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            sentence_count = 0
            line_count = 0
            current_sentence = []
            for line in lines:
                line = line.strip()
                if not line:
                    if current_sentence:
                        f.write('\n'.join('\t'.join(parts) for parts in current_sentence) + '\n\n')
                        sentence_count += 1
                        current_sentence = []
                    continue
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) == 3 and all(parts):
                    current_sentence.append(parts)
                    line_count += 1
                else:
                    logger.warning(f"Skipping malformed line: {line} (parts: {parts})")
            if current_sentence:
                f.write('\n'.join('\t'.join(parts) for parts in current_sentence) + '\n')
                sentence_count += 1
            if sentence_count == 0:
                logger.error("No valid sentences found in the input file.")
                raise ValueError("No valid sentences after fixing file format")
        logger.info(f"Fixed file saved to: {output_file}")
        logger.info(f"Processed {line_count} valid lines in {sentence_count} sentences")
        with open(output_file, 'r', encoding='utf-8') as f:
            logger.info(f"First 5 lines of fixed file:\n{''.join(f.readlines()[:5])}")
    except Exception as e:
        logger.error(f"Failed to fix file format: {e}")
        raise

fix_file_format(FILE, FIXED_FILE)

# 2. Load the new dataset
try:
    new_dataset = load_dataset("text", data_files={"train": FIXED_FILE})
except Exception as e:
    logger.error(f"Failed to load new dataset: {e}")
    raise

# 3. Parse raw lines to sentence-wise format
def parse_lines_to_sentences(dataset_split: List[Dict]) -> List[List[List[str]]]:
    sentences = []
    current_sentence = []
    for i, item in enumerate(dataset_split):
        line = item.get('text', '').strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue
        parts = line.split('\t')
        if len(parts) != 3 or '' in parts:
            logger.warning(f"Skipping malformed line at index {i}: {line}")
            continue
        current_sentence.append(parts)
    if current_sentence:
        sentences.append(current_sentence)
    logger.info(f"Parsed {len(sentences)} sentences")
    if sentences:
        logger.info(f"Sample sentence: {sentences[0][:5]}")
    return sentences

new_train_sentences = parse_lines_to_sentences(new_dataset["train"])
if not new_train_sentences:
    logger.error("No sentences parsed. Check the dataset file format.")
    raise ValueError("Empty dataset after parsing")

# 4. Load existing label mappings
existing_labels = ['B', 'B_BRN', 'B_DES', 'B_DTM', 'B_LOC', 'B_MEA', 'B_NAME', 'B_NUM', 'B_ORG', 'B_PER', 'B_TRM', 'B_TTL', 
                   'DDEM', 'E_BRN', 'E_DES', 'E_DTM', 'E_LOC', 'E_MEA', 'E_NUM', 'E_ORG', 'E_PER', 'E_TRM', 'E_TTL', 
                   'I', 'I_BRN', 'I_DES', 'I_DTM', 'I_LOC', 'I_MEA', 'I_NUM', 'I_ORG', 'I_PER', 'I_TRM', 'I_TTL', 
                   'MEA_BI', 'O', 'OBRN_B', 'ORG_I', 'PER_I', '__']
ner_label2id = {label: i for i, label in enumerate(existing_labels)}
ner_id2label = {i: label for label, i in ner_label2id.items()}

# Check for new labels
def extract_labels(sentences: List[List[List[str]]], column_idx: int) -> List[str]:
    label_set = set()
    for sentence in sentences:
        for word in sentence:
            if len(word) > column_idx:
                label_set.add(word[column_idx])
    return sorted(label_set)

new_ner_labels = extract_labels(new_train_sentences, 2)
new_labels = list(set(existing_labels + new_ner_labels))
if new_labels != existing_labels:
    logger.warning(f"New labels detected: {set(new_ner_labels) - set(existing_labels)}")
    ner_label2id = {label: i for i, label in enumerate(new_labels)}
    ner_id2label = {i: label for label, i in ner_label2id.items()}
    logger.info(f"Updated NER labels: {new_labels}")

# 5. Convert to HuggingFace dataset format
def convert_to_hf_format(sentences: List[List[List[str]]]) -> List[Dict]:
    hf_data = []
    for sentence in sentences:
        tokens = [word[0] for word in sentence]
        ner_tags = [ner_label2id.get(word[2], ner_label2id['O']) for word in sentence]
        hf_data.append({"tokens": tokens, "ner_labels": ner_tags})
    return hf_data

hf_new_train = convert_to_hf_format(new_train_sentences)
new_train_dataset = Dataset.from_list(hf_new_train)
logger.info(f"Dataset size: {len(new_train_dataset)} sentences")
logger.info(f"Sample dataset entry: {new_train_dataset[0]}")

# 6. Tokenization + align labels
tokenizer = AutoTokenizer.from_pretrained("./ner_model_final")
# Add กะโนยโคยโทง to tokenizer vocabulary
new_token = "กะโนยโคยโทง"
if new_token not in tokenizer.vocab:
    tokenizer.add_tokens([new_token])
    logger.info(f"Added {new_token} to tokenizer vocabulary")

def tokenize_and_align_labels(batch: Dict) -> Dict:
    tokens = batch["tokens"]
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], str):
        tokens = [tokens]
    elif isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens
    else:
        logger.error(f"Invalid tokens format: {tokens}")
        raise ValueError("Invalid tokens format in batch")
    logger.info(f"Token input to tokenizer: {tokens[:2]}")

    tokenized_inputs = tokenizer(
        tokens,
        truncation=True,
        max_length=510,
        padding="max_length",
        is_split_into_words=True,
        return_tensors="pt"
    )

    ner_all_labels = []
    for i in range(len(tokens)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        ner_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                ner_labels.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    ner_labels.append(batch["ner_labels"][i][word_idx])
                except IndexError:
                    logger.error(f"IndexError: i={i}, word_idx={word_idx}, ner_labels={batch['ner_labels'][i]}")
                    raise
            else:
                ner_labels.append(-100)
            previous_word_idx = word_idx
        ner_all_labels.append(ner_labels)

    tokenized_inputs["labels"] = torch.tensor(ner_all_labels, dtype=torch.long)
    logger.info(f"Tokenized inputs keys: {list(tokenized_inputs.keys())}")
    logger.info(f"Input IDs shape: {tokenized_inputs['input_ids'].shape}")
    logger.info(f"Labels shape: {tokenized_inputs['labels'].shape}")

    return tokenized_inputs

tokenized_new_train = new_train_dataset.map(tokenize_and_align_labels, batched=True, batch_size=1)
tokenized_new_train = tokenized_new_train.remove_columns(new_train_dataset.column_names)
logger.info(f"Columns after removal: {tokenized_new_train.column_names}")

# 7. Load the pre-trained model
try:
    model = AutoModelForTokenClassification.from_pretrained(
        "./ner_model_final",
        num_labels=len(ner_label2id),
        id2label=ner_id2label,
        label2id=ner_label2id
    )
    # Resize model embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# 8. Define training args for fine-tuning
training_args = TrainingArguments(
    output_dir="./ner_finetuned_model",
    eval_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=1,
    num_train_epochs=20,  # Increased for small dataset
    learning_rate=3e-5,   # Slightly higher to learn new token
    save_steps=500,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False
)

# 9. Fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_new_train,
    processing_class=tokenizer
)

try:
    trainer.train()
except Exception as e:
    logger.error(f"Fine-tuning failed: {e}")
    raise

# 10. Save the fine-tuned model
save_path = "./ner_finetuned_model_final"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
logger.info(f"Fine-tuned model and tokenizer saved to: {save_path}")