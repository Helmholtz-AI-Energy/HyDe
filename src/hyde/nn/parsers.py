from . import models

# import models

__all__ = ["qrnn_parser"]


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def qrnn_parser(parser):
    parser = basic_parser(parser)
    parser.add_argument("--chop", action="store_true", help="forward chop")
    parser.add_argument("--clip", type=float, default=1e6)
    parser.add_argument(
        "--tensorboard", action="store_true", help="log with tensorboard and stdout"
    )
    return parser.parse_args()


def basic_parser(parser):
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        required=True,
        choices=model_names,
        help="model architecture: " + " | ".join(model_names),
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4, help="training batch size. default=16"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate. default=1e-3.")
    parser.add_argument("--wd", type=float, default=0, help="weight decay. default=0")
    parser.add_argument(
        "--loss",
        type=str,
        default="l2",
        help="which loss to choose.",
        # choices=["l1", "l2", "smooth_l1", "ssim", "l2_ssim"],
    )
    parser.add_argument(
        "--nn-init-mode",
        type=str,
        default="kn",
        help="which init scheme to choose.",
        choices=["kn", "ku", "xn", "xu", "edsr"],
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers for data loader to use"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed to use. default=2018")
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    parser.add_argument(
        "--resume-training",
        "-rt",
        action="store_true",
        help="resume training from checkpoint epoch",
    )
    parser.add_argument("--no-resume-opt", "-nro", action="store_true", help="not resume optimizer")
    parser.add_argument("--resume_path", "-rp", type=str, default=None, help="checkpoint to use.")
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
    parser.add_argument(
        "--save-dir",
        "-s",
        type=str,
        default="/data/weikaixuan/hsi/data/",
        help="directory to save model to",
    )
    parser.add_argument("--log-freq", type=int, default=10, help="how frequently to log outputs")
    parser.add_argument("--no-cuda", action="store_true", help="log with tensorboard and stdout")
    parser.add_argument(
        "--comm-method", type=str, default="nccl-mpi", help="how to spawn processes"
    )
    return parser


def benchmark_parser(parser):
    parser.add_argument("--method", type=str, help="Denoisnig method")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--data-dir", type=str, help="location of the noisy images")
    parser.add_argument("--output-dir", type=str, help="location of the output csv")
    parser.add_argument("--original-image", type=str, help="location of the original file")

    return parser.parse_args()
