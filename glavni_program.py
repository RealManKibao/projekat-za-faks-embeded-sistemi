import sounddevice as sd
import time
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperAttention
import queue

#testiram da li je dobro importovan C++ fajl koji sam pisao
try:
    import moj_akcelerator
    print("C++ fajl uspesno uvezen.")
except ImportError:
    print("Nije moguce uvesti C++ fajl. Probaj da kompajliras kod ponovo.")
    exit()

#podesavanje modela
MODEL_NAME = "openai/whisper-small" #small model trenutno koristim, ali moze i jaci
RECORD_SECONDS = 10 #nije uradjeno u real timeu, pa mi je bitno koliko dugo cu snimati
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

q = queue.Queue()   #ovo je nas red koji koristimo za komunkaciju izmedju dela za audio i glavnog programa(main-a)

"""ova funkcija se zove stalno kada se pali mikrofon,
radi tako sto prvo primi neki zvuk sa mikrofona(indata) zatim ga,
stavi u red(queue) kako bi ga main obradio kada na to bude bio spreman,
razlog zbog cega se ulazni podaci kopiraju indata.copy je jer queue voli nekad da obrise audio fajlove iz indata bafera"""
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())    #stavlja indata u queue

#klasa za ceo attention blok    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
class AttentionBlock(WhisperAttention):
    def __init__(self, embed_dim, num_heads, dropout, is_decoder=False):    #konstruktor klase koji dodeljuje pocetne vrednosti i inicijalizuje svaki objekat
        super().__init__(embed_dim, num_heads, dropout, is_decoder)
    
    def forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None, layer_head_mask=None, output_attentions=False):
        print("Pokrece se C++ kod...")

        #koristimo originalne slojeve linear slojeve za vrednosti matrica Q, K i V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        #prebacujemo Q, K i V u NumPy i saljemo nasem C++ fajlu
        q_numpy = Q.detach().cpu().numpy.squeeze(0)
        k_numpy = K.detach().cpu().numpy.squeeze(0)
        v_numpy = V.detach().cpu().numpy.squeeze(0)
        
        #pozivamo moju C++ funckiju koja radi sve
        attention_output_numpy = moj_akcelerator.attention(q_numpy, k_numpy, v_numpy)
        #vracamo rezultat u PyTorch Tenzor
        attention_output = torch.from_numpy(np.array(attention_output_numpy)).unsqueeze(0).to(hidden_states.device, dtype=hidden_states.dtype)
        #out_proj ostaje originalan
        attention_output = self.out_proj(attention_output)
        
        print("C++ funkcija je uspesno izvrsena!")
    
        return (attention_output, None, None)   #ovako vracamo rezultat u formatu koji model ocekuje
    
def main():
    print(f"Ucitavanje modela '{MODEL_NAME}'...\n")
    device = "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    print("\n[INFO] Menjam originalni Whisper 'Attention' sloj našim akcelerisanim slojem...")
    original_layer = model.model.encoder.layers[0].self_attn
    novi_sloj = AttentionBlock(
        embed_dim=original_layer.embed_dim,
        num_heads=original_layer.num_heads,
        dropout=original_layer.dropout
    )
    
    novi_sloj.load_state_dict(original_layer.state_dict())
    model.model.encoder.layers[0].self_attn = novi_sloj
    print("zamena uspesna.\n")

    # Glavna petlja koja radi neprestano
    while True:
        try:
            num_blocks_to_record = int(RECORD_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
            audio_data = []
            
            print("-------------------------------------------")
            input(f"Pritisnite Enter da započnete snimanje od {RECORD_SECONDS} sekundi...")
            
            print(f"Snimam...")
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCK_SIZE, callback=audio_callback):
                for _ in range(num_blocks_to_record):
                    audio_data.append(q.get())
            print("Snimanje završeno, obrađujem snimak...")

            if audio_data:
                full_audio = np.concatenate(audio_data, axis=0).flatten()
                input_features = processor(full_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                print("\n****FINALNA TRANSKRIPCIJA****")
                print(transcription[0].strip() if transcription and transcription[0].strip() else "[Nema prepoznatog govora]")
                print("---------------------------\n")

        except KeyboardInterrupt:
            print("\nProgram prekinut.")
            break
        except Exception as e:
            print(f"Došlo je do greške: {e}")
            break
if __name__ == "__main__":
    main()