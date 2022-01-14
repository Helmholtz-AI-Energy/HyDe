from setuptools import Extension, setup
from torch.utils import cpp_extension

try:
    setup(
        name="split_bregman",
        ext_modules=[cpp_extension.CppExtension(
            "split_bregman",
            ["src/hyde/cpp/split_bregman.cpp"]
        )],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
except:  # noqa
    print("Failed while building split bregmann extension")
