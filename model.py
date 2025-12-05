import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        # embedding, LeNet5-like architecture
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        # attention mechanism, equation (8) in the paper Iise 2018
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),  # tanh(Vh^T)
            nn.Linear(
                self.L, self.ATTENTION_BRANCHES
            ),  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        # classifier, FCN
        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        # embedding
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        # attention mechanism
        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        # equation (7) in the paper Iise 2018
        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), nn.Tanh()  # matrix V
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), nn.Sigmoid()  # matrix U
        )

        self.attention_w = nn.Linear(
            self.L, self.ATTENTION_BRANCHES
        )  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(
            A_V * A_U
        )  # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A


# ==========================================================================
# PyTorch Lightning Modules
# ==========================================================================
class LitAttention(pl.LightningModule):
    """PyTorch Lightning wrapper for Attention model"""

    def __init__(
        self,
        lr: float = 0.0005,
        weight_decay: float = 10e-5,
        no_validation: bool = False,
    ):
        super().__init__()
        self.model = Attention()
        self.lr = lr
        self.weight_decay = weight_decay
        self.no_validation = no_validation
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        bag_label = y[0]
        loss, _ = self.model.calculate_objective(x, bag_label)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.no_validation:
            return None
        x, y = batch
        bag_label = y[0]
        loss, _ = self.model.calculate_objective(x, bag_label)
        error, _ = self.model.calculate_classification_error(x, bag_label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_error", error, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        bag_label = y[0]
        loss, _ = self.model.calculate_objective(x, bag_label)
        error, _ = self.model.calculate_classification_error(x, bag_label)
        self.log("test_loss", loss)
        self.log("test_error", error)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )


class LitGatedAttention(pl.LightningModule):
    """PyTorch Lightning wrapper for GatedAttention model"""

    def __init__(
        self,
        lr: float = 0.0005,
        weight_decay: float = 10e-5,
        no_validation: bool = False,
    ):
        super().__init__()
        self.model = GatedAttention()
        self.lr = lr
        self.weight_decay = weight_decay
        self.no_validation = no_validation
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        bag_label = y[0]
        loss, _ = self.model.calculate_objective(x, bag_label)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.no_validation:
            return None
        x, y = batch
        bag_label = y[0]
        loss, _ = self.model.calculate_objective(x, bag_label)
        error, _ = self.model.calculate_classification_error(x, bag_label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_error", error, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        bag_label = y[0]
        loss, _ = self.model.calculate_objective(x, bag_label)
        error, _ = self.model.calculate_classification_error(x, bag_label)
        self.log("test_loss", loss)
        self.log("test_error", error)

    def predict_step(self, batch, batch_idx):
        data, label = batch
        bag_label = label[0]
        instance_labels = label[1]
        _, predicted_label, attention_weights = self.model.forward(data)
        bag_level = (bag_label.item(), int(predicted_label.item()))
        instance_level = list(
            zip(
                instance_labels.tolist()[0],
                attention_weights.tolist()[0],
            )
        )

        return bag_level, instance_level

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
