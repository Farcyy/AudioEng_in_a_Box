# Audio Engineer in a Box: Using DDSPs in a compact Neural Network

## About
**Audio Engineer in a Box** is a compact neural network for real-time live music enhancement on resource-constrained devices. Our framework integrates lightweight neural network control with differentiable digital signal processing (DDSP) modules. Unlike traditional methods using autoencoders, music signals bypass the need for embedding and reconstruction, allowing for direct processing. This reduces model size and streamlines the inference pipeline. DDSP modules are essential, enabling gradient descent for network training. We train on a synthetic dataset replicating a live music venue's acoustics using impulse responses (IRs) from a club's PA system and smartphone recordings, simulating a concert experience. This approach ensures the network generalizes effectively to real-world scenarios with complex room acoustics and audience noise captured by smartphones.

## Methods
**Audio Engineer in a Box** was built 5 second clips of music clips from [MTG_Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) and [MUSD18](https://doi.org/10.5281/zenodo.1117372). The data is processed to build pairs of simulated 'clean' professional live-music recordings and 'dirty' simulated low-quality live recordings. It then performs a style-transfer task, to enhance the 'dirty' input, given the 'clean' reference data. The model employs **convolution layers**. Two different model architectures were tested.

## Paper
The paper and project were created by Constantin von Estorff and Alexander Krause, audio technology students of TU-Berlin. The paper can be found [here](./AE_in_Box.pdf).

## Try it yourself

Be sure that you have **python 3.10 or higher** and pip. 

Get your local instance by cloning from the [original repo](https://github.com/SanjaKrause/AEinBOX)'s `main` branch via 
```
$ git clone https://github.com/Farcyy/AudioEng_in_a_Box
```
Then set-up the environment, using conda take
```
$ conda create --name AEinBOX python=3.10.12
```
to activate the environment, use
```
$ conda activate AEinBOX
```
 To install all the requirements you can use
```
$ pip install -r requirements.txt
```

**Audio Engineer in a Box** can now be started

### Try it yourself.
In [this](https://drive.google.com/drive/folders/1BdHZkVLN6VWR3sCgKXmthoi0wWcDxmqT?usp=sharing) Google Drive Folder, you can download test data, training-runs for the small and the big model, as well as some example results audio files.

By running [`main.py`], you can start an own training. For this, please put the 'clean' and 'dirty' testdata in [`datasets/test`]. In [own_config.py], you can set training parameters.

To reproduce metrics and test the network with own data of live-recordings, you can use [`metrics_inference.ipynb`] simply provide the path to your test input and reference file, as well as your desired weights and run the cells for the desired output. To date, only 5 second snippets are being accepted, which will be further developed in later versions. Don't forget to adjust the number of hidden dimensions in [own_config.py] dependent on whether you want to test the small (32 hidden dim) oder the big (256 hidden dim) network.
