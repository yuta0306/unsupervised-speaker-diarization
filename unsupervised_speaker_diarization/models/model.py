from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from unsupervised_speaker_diarization.losses import jitter_loss


@dataclass
class OutputDiarization:
    embedding_basis_matrix: torch.Tensor
    activation_matrix: torch.Tensor
    loss: torch.Tensor


class DiarizationModel(nn.Module):
    def __init__(
        self,
        lambda1: float = 0.3366,
        lambda2: float = 0.2424,
        lambda3: float = 0.06,
        lr: float = 1e-1,
        device: str = "cpu",
    ) -> None:
        super(DiarizationModel, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lr = lr
        self.device = device

    def forward(self, embedding: torch.Tensor, k: int, eps: float = 1e-6, patience: int = 5) -> OutputDiarization:
        """
        Parameters
        ----------
        embedding: torch.Tensor
            `(embedding_dim, sequence_length)`
        """
        embedding = embedding.to(self.device)  # (time, dim)
        i = 0
        embedding_basis_matrix = nn.Parameter(
            torch.randn((k, embedding.size(1)), device=self.device), requires_grad=True
        )  # (spk, dim)
        activation_matrix = nn.Parameter(
            torch.randn((k, embedding.size(0)), device=self.device), requires_grad=True
        )  # (spk, time)
        optimizer_emb = optim.Adam(params=[embedding_basis_matrix], lr=self.lr)
        optimizer_act = optim.Adam(params=[activation_matrix], lr=self.lr)

        loss = float("inf")
        best = float("inf")
        best_emb = embedding_basis_matrix
        best_act = activation_matrix
        patience_ = patience
        writer = SummaryWriter()

        while loss > eps:
            # compute loss
            loss = (
                torch.linalg.norm(
                    embedding - torch.matmul(activation_matrix.detach().T, embedding_basis_matrix), ord=1
                )
                + self.lambda1 * torch.linalg.norm(embedding_basis_matrix, ord=1)
                + self.lambda2 * torch.linalg.norm(activation_matrix.detach(), ord=1)
                + self.lambda3 * jitter_loss(activation_matrix.detach())
            )

            optimizer_emb.zero_grad()
            loss.backward()
            # update embedding using Adam
            optimizer_emb.step()

            embedding_basis_matrix = self._shrink(embedding_basis_matrix, l=self.lambda1)
            embedding_basis_matrix = self._project_unitdisk(embedding_basis_matrix)

            embedding_basis_matrix = nn.Parameter(embedding_basis_matrix, requires_grad=True)

            # recompute loss
            loss = (
                torch.linalg.norm(
                    embedding - torch.matmul(activation_matrix.T, embedding_basis_matrix.detach()), ord=1
                )
                + self.lambda1 * torch.linalg.norm(embedding_basis_matrix.detach(), ord=1)
                + self.lambda2 * torch.linalg.norm(activation_matrix, ord=1)
                + self.lambda3 * jitter_loss(activation_matrix)
            )

            # # update activation using Adam
            optimizer_act.zero_grad()
            loss.backward()
            optimizer_act.step()

            activation_matrix = self._shrink(activation_matrix, l=self.lambda2)
            activation_matrix = self._project_0to1(activation_matrix)

            activation_matrix = nn.Parameter(activation_matrix, requires_grad=True)

            writer.add_scalar("loss", loss.item(), global_step=i)
            i += 1

            if loss.item() < best:
                best = loss.item()
                best_emb = embedding_basis_matrix.detach().cpu().numpy()
                best_act = activation_matrix.detach().cpu().numpy()
                patience_ = patience
            else:
                patience_ -= 1
            
            if patience_ < 0:
                break

        return OutputDiarization(
            embedding_basis_matrix=best_emb,
            activation_matrix=best_act,
            loss=best,
        )

    def _shrink(self, X: nn.Parameter, l: float) -> nn.Parameter:
        return torch.sign(X) * torch.max(
            torch.zeros_like(X), torch.abs(X - self.lr * l)  # lagrange ???
        )

    def _project_unitdisk(self, X: nn.Parameter) -> nn.Parameter:
        return X / torch.linalg.norm(X, dim=1, ord=2, keepdim=True)

    def _project_0to1(self, X: nn.Parameter) -> nn.Parameter:
        return torch.max(torch.tensor(0.0), torch.min(torch.tensor(1.0), X))


if __name__ == "__main__":
    from unsupervised_speaker_diarization.models.detect_speech import SpeechDetection
    import librosa
    import warnings
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    warnings.simplefilter("ignore")

    # pre process
    detector = SpeechDetection()
    audio, sr = librosa.load("data/T0iOKreqf2k.mp3", sr=16_000)
    output = detector(audio)
    mask = detector.masked_array(output.scores, type_="speech")
    embedding = output.embeddings
    embedding[~mask.astype(bool)] = 0

    
    model = DiarizationModel(lr=1e-4, device="cuda")
    embedding = torch.from_numpy(embedding)
    output = model(embedding, k=3)
    # print(output.embedding_basis_matrix)
    print(output.loss)

    # post process
    final_act = output.activation_matrix
    final_act[:, ~mask.astype(bool)] = 0
    final_act_shift = np.concatenate([final_act[:, :1], final_act[:, :-1]], axis=1)
    prob = np.mean(np.stack([final_act, final_act_shift]), axis=0)
    mask = prob > 0.2
    final_prob = np.zeros_like(prob)
    final_prob[mask] = 1
    print(final_prob)

    data = np.array([[value for value in spk for _ in range(24)] for spk in final_prob])
    index = [0.020 * (i + 1) for i in range(data.shape[1])]
    df = pd.DataFrame(data=data.T, columns=["spk1", "spk2", "spk3"], index=index)
    df.to_csv("diarization.csv", header=True, index=True)


    spk1 = np.array([p == 1 for p in final_prob[0] for _ in range(int(sr * 0.480))])
    spk2 = np.array([p == 1 for p in final_prob[1] for _ in range(int(sr * 0.480))])
    spk1 = np.concatenate([np.array([final_prob[0][0] == 1] * int(sr * 0.480)), spk1])[:audio.shape[0]]
    spk2 = np.concatenate([np.array([final_prob[1][0] == 1] * int(sr * 0.480)), spk2])[:audio.shape[0]]
    audio1 = audio.copy()
    audio1[~spk1] = 0
    audio2 = audio.copy()
    audio2[~spk2] = 0
    sf.write("spk1.wav", audio1, samplerate=sr)
    sf.write("spk2.wav", audio2, samplerate=sr)

    plt.figure(figsize=(30, 4))
    sns.lineplot(df)
    plt.show()