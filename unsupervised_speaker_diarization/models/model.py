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
        embedding = embedding.to(self.device)
        i = 0
        embedding_basis_matrix = nn.Parameter(
            torch.randn((embedding.size(0), k), device=self.device), requires_grad=True
        )
        activation_matrix = nn.Parameter(
            torch.randn((k, embedding.size(1)), device=self.device), requires_grad=True
        )
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
                    embedding - torch.matmul(embedding_basis_matrix, activation_matrix.detach()), ord=1
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
                    embedding - torch.matmul(embedding_basis_matrix.detach(), activation_matrix), ord=1
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
    model = DiarizationModel(device="cuda")
    embedding = torch.randn((512, 100))
    output = model(embedding, k=3)
    print(output.embedding_basis_matrix)
    print(output.activation_matrix)
    print(output.loss)
