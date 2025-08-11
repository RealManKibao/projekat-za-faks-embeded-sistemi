import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import queue
import time

# --- Podesavanja ---
MODEL_NAME = "openai/whisper-small"
RECORD_SECONDS = 10
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

# Red za komunikaciju
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Greška u audio stream-u: {status}")
    q.put(indata.copy())

def main():
    """Glavna funkcija za testiranje zamene."""
    print(f"Učitavanje modela '{MODEL_NAME}'...")
    device = "cpu"
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        print(f"Greška pri učitavanju modela: {e}")
        return

    print(f"Model učitan i radi na uređaju: {device}")

    # --- POČETAK KODA ZA ZAMENU ---

    # 1. Učitavamo rezultat iz vašeg C++ fajla
    try:
        print("\n[INFO] Učitavam rezultat iz 'izlaz_Y_cpp.txt'...")
        cpp_result_numpy = np.loadtxt("izlaz_Y_cpp.txt")
        # Pretvaramo ga u PyTorch tenzor i prebacujemo na pravi uređaj (cpu)
        cpp_result_tensor = torch.from_numpy(cpp_result_numpy).float().to(device)
        # Vraćamo ga u originalni 3D oblik (ovo je ključno!)
        cpp_result_tensor = cpp_result_tensor.view(1, 1500, 3072)
        print(f"[INFO] C++ rezultat uspešno učitan, dimenzije: {cpp_result_tensor.shape}")
    except FileNotFoundError:
        print("\n[GREŠKA] Fajl 'izlaz_Y_cpp.txt' nije pronađen! Prvo morate pokrenuti vaš C++ program.")
        return

    # 2. Definišemo hook koji će "ubrizgati" naš rezultat
    def injection_hook(module, input, output):
        # 'output' je originalni rezultat koji bi PyTorch izračunao.
        # Mi ga IGNORIŠEMO i umesto njega VRAĆAMO naš tenzor.
        print("\n---------- HOOK AKTIVIRAN: Ubrizgavam C++ podatke! ----------")
        print(f"Originalni PyTorch izlaz bi bio dimenzija: {output.shape}")
        print(f"Vraćam tenzor iz C++ fajla dimenzija: {cpp_result_tensor.shape}")
        
        # Uklanjamo hook da se ne bi aktivirao više puta
        hook_handle.remove()
        
        # Vraćanjem vrednosti, mi menjamo tok izvršavanja modela!
        return cpp_result_tensor

    # 3. Kačimo hook na isti ciljni sloj
    target_layer = model.model.encoder.layers[0].fc1
    hook_handle = target_layer.register_forward_hook(injection_hook)
    print(f"[INFO] Postavljen 'injection hook' na sloj: {target_layer.__class__.__name__}\n")

    # --- KRAJ KODA ZA ZAMENU ---

    # Snimamo isti audio kao i pre da bi ulaz bio identičan
    try:
        # ... (kod za snimanje zvuka je identičan kao pre) ...
        num_blocks_to_record = int(RECORD_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
        audio_data = []
        print("Snimanje počinje (koristimo isti snimak kao za generisanje test podataka)...")
        time.sleep(2)
        print(f"Snimam {RECORD_SECONDS} sekundi...")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCK_SIZE, callback=audio_callback):
            for _ in range(num_blocks_to_record):
                audio_data.append(q.get())
        print("Snimanje završeno, obrađujem snimak sa 'podmetnutim' rezultatom...")

        if audio_data:
            full_audio = np.concatenate(audio_data, axis=0).flatten()
            input_features = processor(full_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            print("\n--- FINALNA TRANSKRIPCIJA (sa C++ rezultatom) ---")
            if transcription and transcription[0].strip():
                print(transcription[0].strip())
            else:
                print("[Model nije prepoznao govor u snimku]")
            print("--------------------------------------------------\n")

    except Exception as e:
        print(f"Došlo je do greške: {e}")

if __name__ == "__main__":
    main()