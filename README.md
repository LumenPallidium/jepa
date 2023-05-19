# JEPA
This repository contains my experiments with Joint Embedding Predictive Architectures (JEPAs). It contains my barebones implementation of [I-JEPA (Image-JEPA)](https://arxiv.org/abs/2301.08243), which is this general architecture applied to the image domain.

## What are JEPAs?

JEPAs are an architecture that utilize self-superivsed learning in order to learn a semantically rich embedding of data. They are predictive in the sense that they are trained to predict the embedding of given portions of data from other embedded portions.

I-JEPA use a framework in which there are three networks: a context encoder, a target encoder, and a predictor. The target encoder is simply an exponentially moving average of the context encoder. The context encoder encodes large portions of an image (using a standard transformer framework, images are broken up into patches and each patch is given an embedding vector in some embedding space). The target encoder, on the other hand, encodes many small chunks of the same image, which are masked from the context portion of the image. The goal of the predictor is to predict these small embedded target chunks from the embedded context chunk. Both the predictor and context encoder are trained via backpropagation and gradient descent.

In this way, the model learns the structure of image, all while working in an embedding space. This is enabled by the fact that the predictor must learn to accurately predict. An advantage of this method compared with something like a (masked) autoencoder is that there is no limitation from a decoder. Not only does a decoder add more parameters to a model, it also adds a task which could in principle be harder than forming a strong representation of a given dataset's manifold (namely, the task of converting an object on that manifold to an object in a measurement space e.g. an image).

## Notes

As an additional note, I'll describe how I intepreted some parts of the paper which I felt left some openness to intepretation (at this point I'm assuming you've looked through the paper). The following image shows a figure from the text describing the functioning of the I-JEPA along with a relevant section of text:

![Description from paper](images/example.png)

For the predictor, I assumed that the input was in the form of a concatenation of a masked target region and the encoded context. Positional information was naturally added via a predictor-specific learnable postional embedding, filtered to only given target and the context.
