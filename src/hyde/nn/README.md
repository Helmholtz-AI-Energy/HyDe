# Notes:
Different NN setups will come with different dataloaders. However, these networks
should work so long as the input data is the expected size.

# Dependencies:
Some packages will use different packages under the hood. The implemented networks should work
as standalone repositories, but they should also have a standard structure and behave similarly.

For example QRNN3D uses `caffe` for loading the data as it uses a LMDB dataset. The model itself
does NOT depend upon this, only the loading of data in the way that they did it

### QRNN3D:
- [Caffe](https://caffe.berkeleyvision.org/installation.html)
