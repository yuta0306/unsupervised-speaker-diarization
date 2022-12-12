import librosa
import tensorflow as tf
import torch

from unsupervised_speaker_diarization.models.detect_speech import SpeechDetection
from unsupervised_speaker_diarization.models.model import DiarizationModel

def load_audio():
    audio, sr = librosa.load("data/T0iOKreqf2k.mp3", sr=16_000)
    # normalize
    # audio = audio / tf.int16.max
    return audio

audio = load_audio()
detector = SpeechDetection()
output = detector(audio)
embeddings = output.embeddings
array = detector.masked_array(output.scores, "speech", threshold=0.01)
print("speech:", array.shape, array)
print(embeddings.shape, embeddings)

model = DiarizationModel(lr=0.01)
output = model(embedding=torch.from_numpy(embeddings), k=3)
print(output)
print(output.activation_matrix.tolist())

# post process
result = output.activation_matrix
import pandas as pd

df = pd.DataFrame(data={
    "speaker 0": result[0].numpy(),
    "speaker 1": result[1].numpy(),
    "speaker 2": result[1].numpy(),
})

df.loc[array == 0] = 0
df.to_csv("test.csv", header=True, index=True)
