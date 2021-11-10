from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name="split_bregman",
    ext_modules=[cpp_extension.CppExtension("split_bregman", ["split_bregman.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
