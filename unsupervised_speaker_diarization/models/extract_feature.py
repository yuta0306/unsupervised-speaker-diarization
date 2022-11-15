from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2Processor


@dataclass
class OutputFeatureExtraction:
    last_hidden_state: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]


class FeatureExtraction:
    """
    HuBERT (https://arxiv.org/pdf/2106.07447.pdf)
    The convolutional waveform encoder generates a feature sequence at a 20ms framerate for audio sampled at 16kHz
    (CNN encoder down-sampling factor is 320x).
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(
            self.device
        )

    def __call__(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> OutputFeatureExtraction:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.Tensor(waveform)

        input_values = self.processor(
            waveform, sampling_rate=16_000, truncation=False, return_tensors="pt"
        ).input_values.to(self.device)
        with torch.no_grad():
            output = self.model(input_values, output_hidden_states=True)

        return OutputFeatureExtraction(
            last_hidden_state=output.last_hidden_state.to("cpu"),
            hidden_states=tuple(
                hidden_state.to("cpu") for hidden_state in output.hidden_states
            ),
        )


if __name__ == "__main__":
    import librosa

    audio, sr = librosa.load("data/_LIDbvp1NYw.mp3", sr=16_000)
    model = FeatureExtraction()
    output = model(audio)
    print(output.last_hidden_state.size())
