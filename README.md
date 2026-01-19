This is X. He's full Python implementation of NeuroSLAM, based on F. Yu's 
original implementation in MATLAB [1]. 

This version has been rigorously tested by saving key data from the raw
MATLAB implementation to evaluate the performance of each Python script, 
thereby assessing its ability to reproduce the NeuroSLAM in MATLAB.

The Python script imresize.py is a reimplementation of MATLAB's built-in
function imresize(), modified from [2].

[1] https://github.com/cognav/NeuroSLAM.git
[2] https://github.com/fatheral/matlab_imresize

Implementation Details
TensorFlow Implementation (for Edge Deployment):

Environment: Python 3.6 / TensorFlow 1.14

Description: This version is optimized for execution on edge devices.

PyTorch Implementation:

Environment: Python 3.9 / PyTorch 1.13

Description: A modern implementation leveraging newer features of the PyTorch ecosystem.
