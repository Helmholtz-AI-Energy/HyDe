"""Create lmdb dataset"""
import os
from typing import Callable, List, Tuple, Union

# from util import *
import numpy as np
from scipy.ndimage import zoom

from hyde.lowlevel import logging, utils

from .. import general_nn_utils

hyde_logger = logging.get_logger(level=20)


def create_lmdb_db_from_mat(
    datadir: str,
    out_db_path_name: str,
    matkey: str,
    ksizes: Union[tuple, list],
    strides: Union[tuple, list],
    load: Callable,
    scales: Union[tuple, list] = (1,),
    crop_sizes: Tuple = None,
    seed: int = 42,
):
    """
    Create database for QRNN3D and save it to a database file of type LMBD.
    Files must be '.mat' files for this function.

    Parameters
    ----------
    datadir: str
        Directory name with the '.mat' files
    out_db_path_name: str
        this should be the path and the name of the database file WITHOUT '.db' at the end
    matkey: str
        the key to use to get the data from the loaded file.
        i.e. `data = scipy.io.loadmat(file)[matkey]`
    ksizes: tuple, list
        sizes to use when converting the data to a volume.
        NOTE: must be the same length as `strides` and `scales`
    strides: tuple, list
        strides to use when converting the data to a volume.
        NOTE: must be the same length as `ksizes` and `scales`
    scales: tuple, list, optional
        The zooms in on the image.
        If more than 1 element is in scales, the zoomed in data will be concatenated as another band.
        NOTE: must be the same length as `strides` and `ksizes`
    load: Callable
        the function which is used to load the data files in `datadir`
    crop_sizes: Tuple, optional
        desired size of the data to be saved. This does a center crop of the raw data
    seed: int, optional
        random seed value
        default: 42

    Returns
    -------
    None
        Data is saved to the specified file
    """
    import caffe  # noqa: E741
    import ipdb  # noqa: E821
    import lmdb  # noqa: E821

    def preprocess(data):
        new_data = []

        data, _ = utils.normalize(data)
        # todo: should this be here????
        # data = np.rot90(data, k=2, axes=(1, 2))  # ICVL
        if crop_sizes is not None:
            data = general_nn_utils.crop_center(data, crop_sizes)

        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            temp = general_nn_utils.lmdb_data_2_volume(
                temp, ksizes=ksizes, strides=list(strides[i])
            )
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)

        return new_data.astype(np.float32)

    file_names = os.listdir(datadir)
    file_names = [fn.split(".")[0] + ".mat" for fn in file_names]

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    # this will fail if it isnt something callable
    data = load(datadir + file_names[0])[matkey]
    data = preprocess(data)

    map_size = data.nbytes * len(file_names) * 1.2

    hyde_logger.debug(f"Data shape: {data.shape}")
    hyde_logger.info(f"Dataset ap size: {map_size / 1024 / 1024 / 1024} GB")

    ipdb.set_trace()
    if os.path.exists(out_db_path_name + ".db"):
        raise Exception(f"database already exists! current name: {out_db_path_name + '.db'}")
    env = lmdb.open(out_db_path_name + ".db", map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(file_names):
            try:
                X = load(datadir + fn)[matkey]
            except Exception as e:
                hyde_logger.warning(f"Loading {datadir + fn} failed!!")
                hyde_logger.warning(f"Exception: \n{e}")
                continue
            X = preprocess(X)
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                # image shape: [..., bands, height, width]
                datum.channels = X.shape[-3]
                datum.height = X.shape[-2]
                datum.width = X.shape[-1]
                datum.data = X[j].tobytes()
                str_id = "{:08}".format(k)
                k += 1
                txn.put(str_id.encode("ascii"), datum.SerializeToString())

            hyde_logger.Debug(f"Finished mat {i}/{len(file_names)} : {fn}")


# Create Pavia Centre dataset
def create_pavia_centre_dataset(datadir):
    from scipy.io import loadmat

    hyde_logger.info("Creating Pavia dataset using scipy.io.loadmat")

    create_lmdb_db_from_mat(
        datadir,
        out_db_path_name="/home/kaixuan/Dataset/PaviaCentre",
        matkey="hsi",  # your own dataset address
        scales=(1,),
        ksizes=(101, 64, 64),
        strides=[(101, 32, 32)],
        crop_sizes=None,
        load=loadmat,
    )


# Create ICVL training dataset
def create_icvl64_31_dataset(datadir):
    import h5py

    hyde_logger.info("Creating icvl64_31 dataset")

    create_lmdb_db_from_mat(
        datadir,
        out_db_path_name="/data/weikaixuan/hsi/data/ICVL64_31",
        matkey="rad",  # your own dataset address
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        crop_sizes=(1024, 1024),
        load=h5py.File,
    )
