import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import queue
import time

# --- GLAVNA PODESAVANJA ---
MODEL_NAME = "openai/whisper-small"
RECORD_SECONDS = 10

# --- Tehnička podesavanja ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

# Red za komunikaciju
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Greška u audio stream-u: {status}")
    q.put(indata.copy())

def main():
    """Glavna funkcija programa."""
    print(f"Učitavanje modela '{MODEL_NAME}'... (Ovo može potrajati pri prvom pokretanju)")
    device = "cpu"
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        print(f"Greška pri učitavanju modela: {e}")
        return

    print(f"Model učitan i radi na uređaju: {device}")

    # =======================================================
    #           KOD ZA INSPEKCIJU (HOOK)
    # =======================================================
    # 1. Izaberimo metu.
    target_layer = model.model.encoder.layers[0].fc1
    print(f"\n[INFO] Postavljam 'hook' na sloj: {target_layer.__class__.__name__}")

    # 2. Definišimo našu "špijunsku" hook funkciju
    def hook_function(module, input, output):
        print("\n---------- HOOK AKTIVIRAN ZA SLOJ: Linear ----------")
        
        ulaz_X = input[0]
        tezine_W = module.weight
        bias_b = module.bias
        izlaz_Y = output
        
        print(f"Dimenzije ULAZA (X):      {ulaz_X.shape}")
        print(f"Dimenzije TEŽINA (W):     {tezine_W.shape}")
        print(f"Dimenzije BIAS-a (b):     {bias_b.shape}")
        print(f"Dimenzije IZLAZA (Y):     {izlaz_Y.shape}")
        
        print("\n[INFO] Čuvam matrice u .txt fajlove za C++ analizu...")
        np.savetxt("ulaz_X.txt", ulaz_X.detach().cpu().numpy().reshape(-1, ulaz_X.shape[-1]))
        np.savetxt("tezine_W.txt", tezine_W.detach().cpu().numpy())
        np.savetxt("bias_b.txt", bias_b.detach().cpu().numpy())
        np.savetxt("izlaz_Y_python.txt", izlaz_Y.detach().cpu().numpy().reshape(-1, izlaz_Y.shape[-1]))
        
        print("[INFO] Fajlovi sačuvani. Uklanjam hook.")
        print("---------------------------------------------------\n")
        hook_handle.remove()

    # 3. Zakačimo našu funkciju na izabrani sloj
    hook_handle = target_layer.register_forward_hook(hook_function)
    # =======================================================
    #           KRAJ KODA ZA INSPEKCIJU
    # =======================================================


    try:
        num_blocks_to_record = int(RECORD_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
        audio_data = []

        print("\n-------------------------------------------")
        print("Hook je postavljen. Snimanje će se aktivirati i sačuvati matrice pri prvoj obradi.")
        for i in range(3, 0, -1):
            print(f"Snimanje počinje za {i}...")
            time.sleep(1)
        
        print(f"Snimam {RECORD_SECONDS} sekundi...")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCK_SIZE, callback=audio_callback):
            for _ in range(num_blocks_to_record):
                audio_data.append(q.get())

        print("Snimanje završeno, obrađujem snimak...")

        if audio_data:
            full_audio = np.concatenate(audio_data, axis=0).flatten()
            
            input_features = processor(full_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
            # Kada se sledeća linija izvrši, naš 'hook' će se aktivirati
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            print("\n--- TRANSKRIPCIJA ---")
            if transcription and transcription[0].strip():
                print(transcription[0].strip())
            else:
                print("[Model nije prepoznao govor u snimku]")
            print("---------------------\n")

    except KeyboardInterrupt:
        print("\nProgram prekinut.")
    except Exception as e:
        print(f"Došlo je do greške: {e}")

if __name__ == "__main__":
    main()