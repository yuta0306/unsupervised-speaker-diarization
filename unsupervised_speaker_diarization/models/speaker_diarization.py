from dataclasses import dataclass
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pyannote.audio import Pipeline

from unsupervised_speaker_diarization.models.detect_speech import SpeechDetection


@dataclass
class OutputSpeakerDiarization:
    pass


class SpeakerDiarization(nn.Module):
    def __init__(
        self,
        model_path: str,
        access_token: str,
        speech_detection: bool = False,
        *args,
        **kwargs,
    ):
        super(SpeakerDiarization, self).__init__()

        self.pipeline = Pipeline.from_pretrained(
            model_path, use_auth_token=access_token, *args, **kwargs
        )
        self.speech_detection = speech_detection
        if speech_detection:
            self.detector = SpeechDetection()

    def forward(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        ext: Optional[str] = None,
        sample_rate: int = 16_000,
        min_speakers: int = 1,
        max_speakers: int = 3,
    ):
        """
        Parameters
        ----------
        audio : str, torch.Tensor, np.ndarray
            Audio file
        """

        discard = False
        if isinstance(audio, str):
            if ext is None:
                ext = audio.split(".")[-1]

            if ext != "wav":
                discard = True
                data, sr = librosa.load(audio, sr=sample_rate)
                if self.speech_detection:
                    output = self.detector(data)
                    data = self.detector.mask_waveform(
                        data,
                        scores=output.scores,
                        type_="speech",
                        threshold=0.005,
                        sr=sr,
                    )
                sf.write("tmp.wav", data=data, samplerate=sr)
                audio = "tmp.wav"
        elif isinstance(audio, torch.Tensor):
            discard = True
            data = audio.numpy()
            if self.speech_detection:
                output = self.detector(data)
                data = self.detector.mask_waveform(
                    data,
                    scores=output.scores,
                    type_="speech",
                    threshold=0.005,
                    sr=sr,
                )
            sf.write("tmp.wav", data=data, samplerate=sample_rate)
            audio = "tmp.wav"
        else:
            discard = True
            if self.speech_detection:
                output = self.detector(data)
                data = self.detector.mask_waveform(
                    data,
                    scores=output.scores,
                    type_="speech",
                    threshold=0.005,
                    sr=sr,
                )
            sf.write("tmp.wav", data=audio, samplerate=sample_rate)
            audio = "tmp.wav"

        diarization = self.pipeline(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
        if discard:
            os.remove(audio)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            sf.write(f"{turn.start:.1f}-{turn.end:.1f}-speaker{speaker}.wav", data[int(sr * turn.start):int(sr * turn.end)], samplerate=sample_rate)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("ACCESS_TOKEN")
    if token is None:
        raise ValueError

    diarization = SpeakerDiarization(
        model_path="pyannote/speaker-diarization",
        access_token=token,
        speech_detection=True,
    )

    diarization("data/_LIDbvp1NYw.mp3", min_speakers=2)
