from .denet import DeNet, DeNetLite3D
from .memnet import MemNet
from .qrnn import QRNNREDC3D, REDC3D, ResQRNN3D

"""Define commonly used architecture"""


def qrnn3d():
    net = QRNNREDC3D(in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def qrnn2d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True, is_2d=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def memnet():
    net = MemNet(31, 64, 6, 6)
    net.use_2dconv = True
    net.bandwise = False
    return net


def hsidenet():   
    #net = DeNet(in_channels=10)
    net = DeNet(in_channels=31)
    net.use_2dconv = True
    net.bandwise = False
    return net

def denet3d():
    net = DeNetLite3D()
    net.use_2dconv = False
    net.bandwise = False

    return net
