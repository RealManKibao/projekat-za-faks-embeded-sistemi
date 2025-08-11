import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperAttention
import queue
import time

#Probamo da uvezemo C++ funkciju.
try:
    import moja_realizacija_funkcije
    print("Uspešno uvezen C++ fajl.")
except ImportError:
    print("Nije moguće uvesti cpp fajl. Da li je uspesno kompajliran program?")
    exit()

# --- Podesavanja ---
MODEL_NAME = "openai/whisper-small"
RECORD_SECONDS = 10 #trajanje snimanja
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

q = queue.Queue()

#Konstruktor koji se stalno zove za dodeljivanje početnih vrednosti objektima.
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

#Attention Block
class AttentionBlock(WhisperAttention):
    def __init__(self, embed_dim, num_heads, dropout, is_decoder=False):
        super().__init__(embed_dim, num_heads, dropout, is_decoder)

    def forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None, layer_head_mask=None, output_attentions=False):
        print("pozivamo Attention Block...\n")

        #Originalni linear slojevi za Q, K i V matrice
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        #Prebacujemo Q, K i V u NumPy format kako bi cpp fajl mogao da razume
        q_numpy = Q.detach().cpu().numpy().squeeze(0)
        k_numpy = K.detach().cpu().numpy().squeeze(0)
        v_numpy = V.detach().cpu().numpy().squeeze(0)
        
        #Cpp fajl koji radi sve
        attn_output_numpy = moja_realizacija_funkcije.attention(q_numpy, k_numpy, v_numpy) 
        #Rezultati se vraćaju nazad u PyTorch tenzor
        attn_output = torch.from_numpy(np.array(attn_output_numpy)).unsqueeze(0).to(hidden_states.device, dtype=hidden_states.dtype)        
        #out projekcija ostaje ista
        attn_output = self.out_proj(attn_output)

        print("Attention Block završio!\n")
        
        return (attn_output, None, None) #Vraćamo rezultat u formatu koji ostatak modela očekuje

#Main funkcija
def main():
    print(f"Učitavanje modela '{MODEL_NAME}'...")
    device = "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    print("\nMenjam originalan WhisperAttention blok sa mojim AttentionBlock...")
    original_layer = model.model.encoder.layers[0].self_attn    #putanja gde se nalazi WhisperAttention
    novi_sloj = AttentionBlock(
        embed_dim=original_layer.embed_dim,
        num_heads=original_layer.num_heads,
        dropout=original_layer.dropout
    )
    novi_sloj.load_state_dict(original_layer.state_dict())
    model.model.encoder.layers[0].self_attn = novi_sloj
    print("Zamena uspešna.\n")

    #Petlja za snimanje koja loopuje
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
                print(transcription[0].strip() if transcription and transcription[0].strip() else "Nema prepoznatog govora")
                print("---------------------------\n")

        except KeyboardInterrupt:
            print("\nProgram prekinut.")
            break
        except Exception as e:
            print(f"Došlo je do greške: {e}")
            break

if __name__ == "__main__":
    main()