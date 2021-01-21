# Image super-resolution-pytorch
This project can quickly process image super-resolution reconstruction at any scale ratio. The feature extractor includes EDSR, RDN, RCAN, maffsrn and IMDN. You can modify the network in the parameter configuration to replace it. At the same time, this project can be used to process grayscale images, you only need to modify some data processing and the number of network input and output channels. In order to better use for C++ configuration, the data processing part of the project uses the cv2 package instead of the commonly used PIL.
If you have any questions, please feel free to communicate

# environments
python3.x

torch1.5.0

cv2

# notice
The arguments of train in configs/config_argument.py, you should change it depend on your setting.

# Train
python train.py

# Test
python test.py, and you should modify your saved models path to test_arguments.model_save_dir in configs/config_argument.py.

# Test single picture
python demo.py, and you should modify your saved models path to parser.model in demo.py.

# Reference
https://github.com/yinboc/liif
