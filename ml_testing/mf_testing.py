import os
import torch
import torch.nn.functional as F

from datasets.MovieLens import MovieLensDataset
from ml_testing.ml_testing import MLTest
from models.matrix_factorization import MatrixFactorization


"""
Instance of MLTest that creates a Matrix Factorization model.
Loads the Movielens dataset for testing.
"""

class MatrixFactorizationTest(MLTest):
    loss = F.mse_loss
    path = './results/mf/model.pth'
    r = 20  # factorization rank

    def __init__(self, load_model=False):
        super().__init__()
        self.train_dataset = MovieLensDataset(train=True)
        self.test_dataset = MovieLensDataset(train=False)
        n_users, n_movies = self.train_dataset.get_dims()
        self.model = MatrixFactorization(n_users, n_movies, self.r)

        if load_model:
            print(f'Loading model from {self.path}')
            state_dict = torch.load(self.path)
            self.model.load_state_dict(state_dict, strict=True)


    def run(self, n_epochs, optimizer, sgd=False, save_model=False, log=False):
        super().run(n_epochs,
                    self.model,
                    optimizer,
                    self.get_train_loader(sgd),
                    self.get_test_loader(),
                    MatrixFactorizationTest.loss,
                    sgd,
                    save_model=save_model,
                    log=log)


    def get_train_loader(self, sgd=False):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1 if sgd else len(self.train_dataset),
            shuffle=sgd
        )


    def get_test_loader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False
        )


    def _save_model(self):
        print(f'Saving model to {self.path}')

        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('results/mf'):
            os.makedirs('results/mf')

        torch.save(self.model.state_dict(), self.path)
