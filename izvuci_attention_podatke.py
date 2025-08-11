import torch
import numpy as np
from transformers import WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-small"
device = "cpu"

def get_attention_data():
    print(f"Učitavanje modela '{MODEL_NAME}'...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval() # Postavljamo model u mod za evaluaciju

    # Uzimamo jedan specifičan 'self-attention' sloj kao našu metu
    target_attn_layer = model.model.encoder.layers[0].self_attn

    print(f"Meta za analizu: {target_attn_layer.__class__.__name__}")

    # Kreiramo lažni ulazni tenzor (dimenzije su iste kao one koje smo videli ranije)
    dummy_input = torch.randn(1, 1500, 768).to(device)

    # --- Ručno izvršavamo korake unutar attention sloja da bismo uhvatili Q, K, V ---

    # 1. Dobijamo Q, K, V množenjem ulaza sa težinskim matricama
    Q = target_attn_layer.q_proj(dummy_input)
    K = target_attn_layer.k_proj(dummy_input)
    V = target_attn_layer.v_proj(dummy_input)

    # 2. Pozivamo ceo 'forward' pass da dobijemo finalni izlaz kao referencu
    #    Ovde je ispravljena linija sa dve donje crte za ignorisanje
    final_output, _, _ = target_attn_layer(hidden_states=dummy_input)


    print("Dimenzije Q:", Q.shape)
    print("Dimenzije K:", K.shape)
    print("Dimenzije V:", V.shape)
    print("Dimenzije finalnog izlaza:", final_output.shape)

    # Čuvamo sve u fajlove
    print("\n[INFO] Čuvam matrice Q, K, V i finalni izlaz u .txt fajlove...")
    # Uklanjamo prvu dimenziju (batch size = 1) da bismo imali 2D matrice
    np.savetxt("ulaz_Q.txt", Q.detach().cpu().numpy().squeeze(0))
    np.savetxt("ulaz_K.txt", K.detach().cpu().numpy().squeeze(0))
    np.savetxt("ulaz_V.txt", V.detach().cpu().numpy().squeeze(0))
    np.savetxt("izlaz_Attention_python.txt", final_output.detach().cpu().numpy().squeeze(0))
    
    print("[INFO] Fajlovi uspešno sačuvani.")

if __name__ == "__main__":
    get_attention_data()