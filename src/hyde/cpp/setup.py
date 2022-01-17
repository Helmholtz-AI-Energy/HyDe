from setuptools import Extension, setup
from torch.utils import cpp_extension

# try:
setup(
    name="split_bregman_cpp",
    ext_modules=[
        cpp_extension.CppExtension("split_bregman_cpp", ["src/hyde/cpp/split_bregman.cpp"])
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
# except:  # noqa
#     print("Failed while building split bregmann extension")


# setup(name='lltm_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})
