# Face recognition using PyTorch and pretrained Inception Res V1
This repo contains face recognition script written with Pytorch.
Tha algorithm is based on the pretrained Inception Res V1 model, [details here](https://github.com/timesler/facenet-pytorch).
I also wrote a [story on Medium]().

The main concept are:
1. Generates face samples
2. Load dataset and train only the last linear layer of Inception Res V1
3. Test model and visualize examples from validations set