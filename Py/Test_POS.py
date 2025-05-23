from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import logging
from typing import List, Dict
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load the saved NER model and tokenizer
# model_path = "./ner_model_final"
model_path = "./ner_finetuned_model_final"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    logger.info("NER model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load NER model or tokenizer: {e}")
    raise

# 2. Define NER label mappings (must match training)
labels = ['B', 'B_BRN', 'B_DES', 'B_DTM', 'B_LOC', 'B_MEA', 'B_NAME', 'B_NUM', 'B_ORG', 'B_PER', 'B_TRM', 'B_TTL', 
          'DDEM', 'E_BRN', 'E_DES', 'E_DTM', 'E_LOC', 'E_MEA', 'E_NUM', 'E_ORG', 'E_PER', 'E_TRM', 'E_TTL', 
          'I', 'I_BRN', 'I_DES', 'I_DTM', 'I_LOC', 'I_MEA', 'I_NUM', 'I_ORG', 'I_PER', 'I_TRM', 'I_TTL', 
          'MEA_BI', 'O', 'OBRN_B', 'ORG_I', 'PER_I', '__']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 3. Define a function to predict NER and POS tags
def predict_ner_and_pos(text: str, model, tokenizer, id2label: Dict[int, str]) -> List[Dict[str, str]]:
    # Segment the Thai text into tokens
    tokens = word_tokenize(text, engine="newmm")
    if not tokens:
        logger.warning("Input text is empty after tokenization")
        return []

    logger.info(f"Segmented tokens: {tokens}")

    # --- NER Prediction ---
    # Tokenize the input for NER
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=510,  # Reserve space for special tokens
        padding="max_length",
        add_special_tokens=True,
        return_tensors=None
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

    # Run NER inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_len, num_labels)

    # Get predicted NER label IDs
    ner_predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    # Align NER predictions with original tokens
    aligned_ner_predictions = []
    previous_word_idx = None
    for i, (word_idx, pred_id) in enumerate(zip(word_ids, ner_predictions)):
        if word_idx is None:
            continue  # Skip special tokens
        if word_idx != previous_word_idx:
            label = id2label.get(pred_id, "UNKNOWN")
            aligned_ner_predictions.append(label)
        previous_word_idx = word_idx

    # --- POS Tagging ---
    pos_predictions = pos_tag(tokens, engine="perceptron")

    # Combine NER and POS predictions
    combined_predictions = [
        {"token": token, "ner_label": ner_label, "pos_label": pos_label[1]}
        for token, ner_label, pos_label in zip(tokens, aligned_ner_predictions, pos_predictions)
    ]

    return combined_predictions

# 4. Test the model
# test_sentence = "เมื่อเวลา 10.30 น. วันที่ 9 สิงหาคม 2548 ที่ว่าการอำเภอจะแนะ จังหวัดนราธิวาส พ.ต.สุคนธรัตน์ ชาวพงษ์ รองผู้บังคับกองพัน ร.132 หน่วยเฉพาะกิจ 35 ปฏิบัติการในพื้นที่ อำเภอจะแนะรับมอบตัวแกนนำและแนวร่วมก่อความไม่สงบจากบ้านละหาร หมู่ที่ 3 ตำบลจะแนะ อำเภอจะแนะ เพื่อแสดงตัวเป็นผู้บริสุทธิ์ เพิ่มอีก 2 คน คือ นายซือตี บาโด อายุ 29 ปี"
# test_sentence = "ที่ตั้ง ตำบลปากตะโก อำเภอทุ่งตะโก จังหวัดชุมพร   ที่มาของชื่อปากน้ำตะโก มาจากชื่อคลองโก และชื่อต้นตะโก เป็นพื้นที่ที่ปากน้ำตะโกไหลลงสู่ทะเล บริเวณใกล้เคียงเป็นชุมชนชาวประมงพื้นบ้าน ชาวบ้านประกอบอาชีพเกษตรกรรมและประมงควบคู่กันไป นอกชายฝั่งมีเกาะน้อยใหญ่กว่า 10 เกาะ บางเกาะมีการทำสัมปทานรังนกอยู่ด้วย โดยชุมชนบริเวณนี้เป็นชาวไทย-จีน เช่นเดียวกับที่ปากน้ำชุมพร ดังปรากฏศาลเจ้าจีนอยู่ด้วย ในพื้นที่มีเรื่องราวเกี่ยวกับ “ไอ้ด่างบางมุด” ซึ่งเป็นจระเข้กินคน เข้ามากินคนบริเวณคลองปากน้ำตะโกถึงคลองบางมุด"
test_sentence = "กะโนยโคยโทง"
logger.info(f"Test sentence: {test_sentence}")

try:
    predictions = predict_ner_and_pos(test_sentence, model, tokenizer, id2label)
    logger.info("Predictions:")
    print(f"{'Token':<15} {'NER':<10} {'POS':<10}")
    print("-" * 35)
    for pred in predictions:
        print(f"{pred['token']:<15} {pred['ner_label']:<10} {pred['pos_label']:<10}")
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    raise