import torch
from espnet2.bin.enh_inference import SeparateSpeech
from espnet_model_zoo.downloader import ModelDownloader


class SpeakerSeparation:
    def __init__(
        self,
        model_name: str = "espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw",
        sr: int = 16_000,
        device: str = "cpu",
    ) -> None:
        d = ModelDownloader()
        cfg = d.download_and_unpack(model_name)
        self.model = SeparateSpeech(
            train_config=cfg["train_config"],
            model_file=cfg["model_file"],
            # for segment-wise process on long speech
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=4,
            normalize_output_wav=True,
            device=device,
        )
        self.sr = sr

    def __call__(self, inputs: torch.Tensor):
        wav = self.model(inputs, self.sr)

        return wav


if __name__ == "__main__":
    import librosa
    import soundfile as sf

    audio, sr = librosa.load("data/_LIDbvp1NYw.mp3", sr=22_050)

    model = SpeakerSeparation(
        model_name="espnet/chenda-li-wsj0_2mix_enh_train_enh_conv_tasnet_raw_valid.si_snr.ave",
        sr=sr,
    )
    audio_enh = model(audio[None, ...])

    sf.write("enh.wav", audio_enh[0].squeeze(), sr)
    sf.write("enh1.wav", audio_enh[1].squeeze(), sr)
