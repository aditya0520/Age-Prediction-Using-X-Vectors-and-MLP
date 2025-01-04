import pandas as pd
import torchaudio
import numpy as np
import os
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import wandb
import numpy as np
import torchaudio
from speechbrain.inference import EncoderClassifier 

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)

wandb.login(key="7664f3b17a98ffe7c64b549e349123b61a9d3024") 

wandb.init(project="x-vector-feature-extraction", config={
    "sample_rate": 8000,
    "num_workers": os.cpu_count() - 1,
})

file_path = '/ocean/projects/cis240138p/aaayush/fisher_segmented_audio/train.xlsx' 
df = pd.read_excel(file_path, sheet_name='Sheet1')

file_pin_dict = df.set_index('File Name')['PIN'].to_dict()

def process_file(args):
    """Process a single file, extracting X-Vector features and saving them."""
    filename, pin, audio_dir, output_dir, sample_rate = args
    wav_path = os.path.join(audio_dir, str(pin), f"{filename}.wav")
    current_output_dir = os.path.join(output_dir, str(pin))
    os.makedirs(current_output_dir, exist_ok=True)

    try:
        signal, fs = torchaudio.load(wav_path)
        
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
        embeddings = classifier.encode_batch(signal)
        x_vector = embeddings.squeeze().numpy()

        # Save the features
        feature_file = os.path.join(current_output_dir, f"{filename}.npy")
        np.save(feature_file, x_vector)

    except Exception as e:
        error_message = f"Error processing file {wav_path}: {e}\n{traceback.format_exc()}\n"
        with open("error_log.txt", "a") as log_file:
            log_file.write(error_message)
        return None 

    finally:
        gc.collect()
    return filename 

def preprocess_and_extract_features_parallel(audio_dir, output_dir, file_pin_dict, sample_rate=8000, num_workers=4):
    """Preprocess audio files and extract X-vector features in parallel."""
    os.makedirs(output_dir, exist_ok=True)

    args = [
        (filename, pin, audio_dir, output_dir, sample_rate)
        for filename, pin in file_pin_dict.items()
    ]

    completed_files = 0
    total_files = len(args)
    log_interval = max(1, total_files // 100) 

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, arg) for arg in args]

        with tqdm(total=total_files, desc="Processing audio files", unit="file") as pbar:
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None: 
                    completed_files += 1
                    pbar.update(1)

                if (i + 1) % log_interval == 0 or completed_files == total_files:
                    wandb.log({
                        "completed_files": completed_files,
                        "total_files": total_files,
                        "progress": completed_files / total_files
                    })

    print(f"X-Vector feature extraction complete. Files saved to {output_dir}")

# Usage
audio_dir = "/ocean/projects/cis240138p/aaayush/fisher_segmented_audio/train"
output_dir = "/ocean/projects/cis240138p/aaayush/fisher_x-vector/train"

num_workers = os.cpu_count() - 1 

preprocess_and_extract_features_parallel(audio_dir, output_dir, file_pin_dict, num_workers=num_workers)

# Finish wandb run
wandb.finish()
