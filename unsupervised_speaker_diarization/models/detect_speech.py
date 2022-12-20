from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


@dataclass
class OutputSpeechDetection:
    scores: np.ndarray
    embeddings: np.ndarray
    spectrograms: np.ndarray


class SpeechDetection:
    """
    YAMNet
    https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
    These 96x64 patches are then fed into the Mobilenet_v1 model to yield a 3x2 array of activations for 1024 kernels\
    at the top of the convolution. These are averaged to give a 1024-dimension embedding,\
    then put through a single logistic layer to get the 521 per-class output scores corresponding to the 960 ms input waveform segment.\
    (Because of the window framing, you need at least 975 ms of input waveform to get the first frame of output scores.)
    """

    def __init__(self) -> None:
        self.backbone = hub.load("https://tfhub.dev/google/yamnet/1")
        self._init_device()

    def _init_device(self) -> None:
        tf.debugging.set_log_device_placement(True)

    def __call__(self, waveform: np.ndarray) -> OutputSpeechDetection:
        scores, embeddings, spectrograms = self.backbone(waveform)

        return OutputSpeechDetection(
            scores=scores.numpy(),
            embeddings=embeddings.numpy(),
            spectrograms=spectrograms.numpy(),
        )

    @staticmethod
    def masked_array(
        scores: np.ndarray, type_: str, threshold: float = 0.01
    ) -> np.ndarray:
        idx = 0 if type_ == "speech" else 13
        array = np.zeros(scores.shape[0], dtype=np.int16)
        mask = scores[:, idx] >= threshold
        array[mask] = 1

        return array

    @staticmethod
    def mask_waveform(
        waveform: np.ndarray,
        scores: np.ndarray,
        type_: str,
        except_type: Optional[int] = None,  # Music 132
        threshold: float = 0.01,
        sr: int = 16_000,
    ):
        idx = 0 if type_ == "speech" else 13
        array = np.zeros(scores.shape[0], dtype=np.int16)
        mask = scores[:, idx] >= threshold
        array[mask] = 1

        if except_type is not None:
            array[scores[:, except_type] > scores[:, idx]] = 0

        mask_waveform = []
        for i, mask in enumerate(array):
            if i == 0:
                mask_waveform += [mask] * int(sr * 0.960)
            else:
                mask_waveform += [mask] * int(sr * 0.480)
        mask = np.array(mask_waveform).astype(bool)

        waveform[~mask[: waveform.shape[0]]] = 0

        return waveform


if __name__ == "__main__":
    import librosa

    audio, sr = librosa.load("data/_LIDbvp1NYw.mp3", sr=16_000)
    print(audio.shape)
    model = SpeechDetection()
    output = model(audio)
    # print(output.scores, output.scores.shape)
    print(output.scores.shape)
    # print(output.embeddings.shape)
    # print(output.spectrograms.shape)

    array = model.masked_array(output.scores, "speech", threshold=0.005)
    print("speech:", array)
    array = model.masked_array(output.scores, "laughter", threshold=0.005)
    print("laughter:", array)

    waveform = model.mask_waveform(
        audio, output.scores, "speech", threshold=0.005, sr=sr
    )
    print(waveform)
