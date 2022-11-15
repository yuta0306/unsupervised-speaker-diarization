import librosa
import tensorflow as tf
import tensorflow_hub as hub


def load_audio():
    audio, sr = librosa.load("data/_LIDbvp1NYw.mp3", sr=16_000)
    # normalize
    audio = audio / tf.int16.max
    return audio


def test_yamnet():
    audio = load_audio()
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    _, embedding, _ = model(audio)
