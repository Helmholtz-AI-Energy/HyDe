import numpy as np
import torch

from ...lowlevel import utils
from . import models

__all__ = ["train_argparse"]


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def train_argparse(parser):
    def _parse_str_args(args):
        str_args = args.split(",")
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args

    parser.add_argument("--prefix", "-p", type=str, default="denoise", help="prefix")
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        required=True,
        choices=model_names,
        help="model architecture: " + " | ".join(model_names),
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=16, help="training batch size. default=16"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate. default=1e-3.")
    parser.add_argument("--wd", type=float, default=0, help="weight decay. default=0")
    parser.add_argument(
        "--loss",
        type=str,
        default="l2",
        help="which loss to choose.",
        choices=["l1", "l2", "smooth_l1", "ssim", "l2_ssim"],
    )
    parser.add_argument(
        "--init",
        type=str,
        default="kn",
        help="which init scheme to choose.",
        choices=["kn", "ku", "xn", "xu", "edsr"],
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="number of workers for data loader to use"
    )
    parser.add_argument("--seed", type=int, default=2018, help="random seed to use. default=2018")
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    parser.add_argument("--no-ropt", "-nro", action="store_true", help="not resume optimizer")
    parser.add_argument("--chop", action="store_true", help="forward chop")
    parser.add_argument("--resumePath", "-rp", type=str, default=None, help="checkpoint to use.")
    parser.add_argument(
        "--datadir",
        "-d",
        type=str,
        default="/data/weikaixuan/hsi/data/ICVL64_31.db",
        help="data directory",
    )
    parser.add_argument(
        "--val-datadir",
        "-v",
        type=str,
        default="/data/weikaixuan/hsi/data/",
        help="validation data directory",
    )
    parser.add_argument("--clip", type=float, default=1e6)
    parser.add_argument(
        "--save-freq", type=int, default=10, help="how frequently to save the model"
    )
    parser.add_argument("--log-freq", type=int, default=10, help="how frequently to log outputs")
    parser.add_argument(
        "--tensorboard", action="store_true", help="log with tensorboard and stdout"
    )
    opt = parser.parse_args()
    return opt


def Visualize3D(data, meta=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    data = np.squeeze(data)

    # for ch in range(data.shape[0]): -> data[ch, ...]
    data = utils.normalize(data, by_band=True, band_dim=0)

    print(np.max(data), np.min(data))

    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    # l = plt.imshow(data[frame,:,:])

    imshow = plt.imshow(data[frame, :, :], cmap="gray")  # shows 256x256 image, i.e. 0th frame
    # plt.colorbar()
    axcolor = "lightgoldenrodyellow"
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, "Frame", 0, data.shape[0] - 1, valinit=0)

    def update(val):
        frame = int(np.around(sframe.val))
        imshow.set_data(data[frame, :, :])
        if meta is not None:
            axframe.set_title(meta[frame])

    sframe.on_changed(update)

    plt.show()
