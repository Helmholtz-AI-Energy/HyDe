import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from .. import general_nn_utils

__all__ = [
    "LMDBDataset",
]


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1, transform=None, target_transform=None, max_readers=None):
        import caffe  # noqa: E821
        import lmdb  # noqa: E821

        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            max_readers=max_readers,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        self.repeat = repeat

        if transform is None:
            transform = transforms.RandomApply(
                [
                    general_nn_utils.RandRot90Transform(),
                    transforms.RandomVerticalFlip(p=0.5),
                ],
                p=0.75,
            )

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        index = index % self.length
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get("{:08}".format(index).encode("ascii"))

        datum = caffe.proto.caffe_pb2.Datum()  # noqa: F821
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        # flat_x = np.fromstring(datum.data, dtype=np.float64)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        if self.transform:
            x = self.transform(x)
        # if self.target_transform:
        #     x = self.transform(x)
        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"
