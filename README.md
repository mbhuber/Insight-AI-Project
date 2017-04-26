# Artifical Face Aging with ACGANs
Insight Artificial Intelligence Project (Spring 2017)

## Requirements:
Python 2.7, Keras 1.2.1 (transitionng to 2.1), Tensorflow 1.0
numpy, OpenCV 2


## References

Papers:

Antipov et al. 2017 https://arxiv.org/abs/1702.01983

Arjovsky et al. 2017 https://arxiv.org/abs/1701.07875

Odena et al. 2016 https://arxiv.org/abs/1610.09585

Radford et al. 2016 https://arxiv.org/abs/1511.06434

Github:

https://github.com/carpedm20/DCGAN-tensorflowmen

https://github.com/bobchennan/Wasserstein-GAN-Keras

https://github.com/buriburisuri/ac-gan

Data:

https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/


## Motivation
Artifical face aging is the ability of aging an given face image to a given age. Instead of using an approach based on computer vision, this project explores Generative Adverserial Networks (GANs) to solve this problem. In addition, different levels of pre-processing are compared how they impact the image quality. 

## Methods
Auxiliary Constrained GAN (ACGAN) with a deep convolutional disrciminator and generator (DCGAN) was implemented and tested. For loss functions, the KL divergence and the Wasserstein Distance were considered.









