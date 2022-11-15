from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from unsupervised_speaker_diarization.losses import jitter_loss


@dataclass
class OutputDiarization:
    embedding_basis_matrix: torch.Tensor
    activation_matrix: torch.Tensor


class DiarizationModel(nn.Module):
    def __init__(
        self,
        lambda1: float = 0.3366,
        lambda2: float = 0.2424,
        lambda3: float = 0.06,
        lr: float = 1e-3,
    ) -> None:
        super(DiarizationModel, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lr = lr

    def forward(self, embedding: torch.Tensor, k: int) -> OutputDiarization:
        """
        Parameters
        ----------
        embedding: torch.Tensor
            `(embedding_dim, sequence_length)`
        """
        i = 0
        embedding_basis_matrix = nn.Parameter(
            torch.randn((embedding.size(0), k)), requires_grad=True
        )
        activation_matrix = nn.Parameter(
            torch.randn((k, embedding.size(1))), requires_grad=True
        )
        optimizer_emb = optim.Adam(params=[embedding_basis_matrix], lr=self.lr)
        optimizer_act = optim.Adam(params=[activation_matrix], lr=self.lr)

        while True:
            # compute loss
            loss = (
                torch.norm(
                    embedding - torch.matmul(embedding_basis_matrix, activation_matrix)
                )
                + self.lambda1 * torch.norm(embedding_basis_matrix)
                + self.lambda2 * torch.norm(activation_matrix)
                + self.lambda3 * jitter_loss(activation_matrix)
            )
            tmp = embedding_basis_matrix.clone()
            # calculate gradient of L w.r.t embedding_basis_matrix, \delta L;
            loss.backward()
            # update embedding using Adam
            optimizer_emb.step()
            embedding_basis_matrix = (
                embedding_basis_matrix - embedding_basis_matrix.grad * self.lr
            )

            embedding_basis_matrix = self._shrink(embedding_basis_matrix)
            break

        return OutputDiarization(
            embedding_basis_matrix=embedding_basis_matrix,
            activation_matrix=activation_matrix,
        )

    def _shrink(self, X: nn.Parameter) -> nn.Parameter:
        return torch.sign(X) * torch.max(
            torch.zeros_like(X), torch.abs(X - self.lr * 1)  # lagrange ???
        )

    def _project_unitdisk(self, X: nn.Parameter) -> nn.Parameter:
        raise NotImplementedError


if __name__ == "__main__":
    model = DiarizationModel()
    embedding = torch.randn((512, 100))
    output = model(embedding, k=3)
    print(output)
