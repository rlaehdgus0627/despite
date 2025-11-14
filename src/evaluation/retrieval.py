import torch
from src.dataset import collate

def encode_text_tmr(model, inputs):
    encoded = model(inputs)
    dists = encoded.unbind(1)
    z = dists[0]
    return z

def compute_text_similarity(text1, text2):
    return torch.mm(text1.unsqueeze(0), text2.unsqueeze(0).T).item()  # Cosine similarity

#encode_text_with = "TMR"
def encode_text_with_tmr(text_emb_model, text_encoder, text):
    with torch.no_grad():
        emb = encode_text_tmr(text_encoder, collate.collate_x_dict([text_emb_model(text)]))
    return emb

def encode_text_list_with_tmr(text_emb_model, text_encoder, text):
    text_tokem_embs = text_emb_model(text)
    text_input = collate.collate_x_dict([{"x" : x, "length" : y} for x, y in zip(text_tokem_embs["x"], text_tokem_embs["length"])], device="cuda")
    with torch.no_grad():
        emb_text = encode_text_tmr(text_encoder, text_input)
    return emb_text

