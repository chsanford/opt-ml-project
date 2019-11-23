import os
import pandas as pd
import torch
import torchvision


"""
Modeled after MNIST dataset: https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html
Downloads and pre-processes the data into train/test tensors.
"""
class MovieLensDataset(torch.utils.data.Dataset):
    url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    training_file = 'train.pt'
    test_file = 'test.pt'
    dim_file = 'dim.pt'
    processed_folder = 'ml-latest-small'
    data_dir = './data'
    train_frac = 0.8
    random_state = 1
    _n_users = None
    _n_movies = None

    def __init__(self, train=True):
        super().__init__()
        self.train = train  # train or test set

        if not self._check_exists():
            self.download()
            self.train_test_split()

        if train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        t = torch.load(os.path.join(self.data_dir, self.processed_folder, data_file))
        self.data, self.targets = t[:, :-1], t[:, -1].float()  # data = (userId, movieId), targets = rating


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index], self.targets[index]


    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_dir, self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.data_dir, self.processed_folder, self.training_file)))


    # Returns the dimensions of the matrix from the data to factorize.
    def get_dims(self):
        if self._n_users is None:
            dim = torch.load(os.path.join(self.data_dir, self.processed_folder, self.dim_file))
            self._n_users = int(dim[0])
            self._n_movies = int(dim[1])
        return self._n_users, self._n_movies

    # Preprocess data
    def train_test_split(self):
        df = pd.read_csv(os.path.join(self.data_dir, self.processed_folder, 'ratings.csv')).drop(columns='timestamp')
        self._n_users = df['userId'].nunique()
        self._n_movies = df['movieId'].nunique()
        df['userId'] = df['userId'] - 1
        df['movieId'] = df['movieId'].rank(method='dense') - 1  # renumber to be consecutive
        train_df = df.sample(frac=self.train_frac, random_state=self.random_state)
        test_df = df.drop(train_df.index)
        # remove users/movies in test set that aren't in the train set
        test_df = test_df[test_df['userId'].isin(train_df['userId'])]
        test_df = test_df[test_df['movieId'].isin(train_df['movieId'])]

        with open(os.path.join(self.data_dir, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(torch.tensor(train_df.to_numpy(dtype=int)), f)
        with open(os.path.join(self.data_dir, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(torch.tensor(test_df.to_numpy(dtype=int)), f)
        with open(os.path.join(self.data_dir, self.processed_folder, self.dim_file), 'wb') as f:
            torch.save(torch.tensor([self._n_users, self._n_movies]), f)


    def download(self):
        if not os.path.exists(os.path.join(self.data_dir, self.processed_folder)):
            torchvision.datasets.utils.download_and_extract_archive(self.url, self.data_dir, remove_finished=True)
