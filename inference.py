import torch
import numpy as np
import xgboost as xgb
import json
from transformers import DistilBertTokenizer, DistilBertModel

# Load artifacts
MODEL_PATH = "artifacts/xgb_fake_news_classifier.json"
TOKENIZER_PATH = "artifacts/distilbert_tokenizer"
LABEL_MAP_PATH = "artifacts/label_map.json"

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)

# Load DistilBERT model (for embeddings)
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()

# Load XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_PATH)

# Load label map
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def get_embedding(text: str):
    """Convert text to DistilBERT CLS embedding"""
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding

def predict_fake_news(text: str):
    """Run text through pipeline → Fake/Real"""
    embedding = get_embedding(text)
    pred = xgb_model.predict(embedding)[0]
    return label_map[str(pred)]
