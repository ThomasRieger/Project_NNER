from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import logging
from typing import List, Dict
from pythainlp.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load the saved model and tokenizer
model_path = "./ner_model_final"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {e}")
    raise

# 2. Define label mappings (must match training)
labels = ['B', 'B_BRN', 'B_DES', 'B_DTM', 'B_LOC', 'B_MEA', 'B_NAME', 'B_NUM', 'B_ORG', 'B_PER', 'B_TRM', 'B_TTL', 
          'DDEM', 'E_BRN', 'E_DES', 'E_DTM', 'E_LOC', 'E_MEA', 'E_NUM', 'E_ORG', 'E_PER', 'E_TRM', 'E_TTL', 
          'I', 'I_BRN', 'I_DES', 'I_DTM', 'I_LOC', 'I_MEA', 'I_NUM', 'I_ORG', 'I_PER', 'I_TRM', 'I_TTL', 
          'MEA_BI', 'O', 'OBRN_B', 'ORG_I', 'PER_I', '__']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 3. Define a function to tokenize and predict
def predict_ner(text: str, model, tokenizer, id2label: Dict[int, str]) -> List[Dict[str, str]]:
    # Segment the Thai text into tokens
    tokens = word_tokenize(text, engine="newmm")
    if not tokens:
        logger.warning("Input text is empty after tokenization")
        return []

    logger.info(f"Segmented tokens: {tokens}")

    # Tokenize the input without converting to tensors immediately
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=510,  # Reserve space for special tokens
        padding="max_length",
        add_special_tokens=True,
        return_tensors=None  # Keep as BatchEncoding
    )

    # Extract word_ids for alignment
    word_ids = encoded.word_ids(batch_index=0)
    if word_ids is None:
        logger.error("word_ids not available in tokenizer output")
        raise ValueError("Tokenizer did not return word_ids")

    # Convert to tensors for model input
    inputs = {key: torch.tensor([encoded[key]], dtype=torch.long) for key in encoded if key not in ["word_ids"]}
    
    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_len, num_labels)

    # Get predicted label IDs
    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()  # Shape: (seq_len,)

    # Align predictions with original tokens
    aligned_predictions = []
    previous_word_idx = None
    for i, (word_idx, pred_id) in enumerate(zip(word_ids, predictions)):
        if word_idx is None:
            continue  # Skip special tokens
        if word_idx != previous_word_idx:
            # Only take the prediction for the first subword of each token
            label = id2label.get(pred_id, "UNKNOWN")
            aligned_predictions.append({"token": tokens[word_idx], "label": label})
        previous_word_idx = word_idx

    return aligned_predictions

# 4. Test the model
test_sentence = "ประธานคณะกรรมการ 40 ปี 14 ตุลา เพื่อประชาธิปไตยสมบูรณ์"
logger.info(f"Test sentence: {test_sentence}")

try:
    predictions = predict_ner(test_sentence, model, tokenizer, id2label)
    logger.info("Predictions:")
    for pred in predictions:
        print(f"Token: {pred['token']:<15} Label: {pred['label']}")
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    raise