import torch
import torch.nn.functional as F

from datasets.MovieLens import MovieLensDataset
from ml_testing.ml_testing import MLTest
from models.matrix_factorization import MatrixFactorization


class MatrixFactorizationTest(MLTest):
    loss = F.mse_loss

    def __init__(self):
        super().__init__()
        self.train_dataset = MovieLensDataset(train=True)
        self.test_dataset = MovieLensDataset(train=False)
        n_users, n_movies = self.train_dataset.get_dims()
        self.r = 20  # factorization rank
        self.model = MatrixFactorization(n_users, n_movies, self.r)

    def run(self, n_epochs, optimizer, sgd=False):
        super().run(n_epochs,
                    optimizer,
                    self.get_train_loader(sgd),
                    self.get_test_loader(),
                    self.model,
                    MatrixFactorizationTest.loss,
                    sgd)

    def get_train_loader(self, sgd=False):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1 if sgd else len(self.train_dataset),
            shuffle=True
        )

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=True
        )

    def visualize_data(self):
        pass
