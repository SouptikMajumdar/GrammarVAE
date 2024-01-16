# Variational Autoencoders that are contrained by a context-free Grammar

![](img/overview.png)

## Overview

![](img/encoder.png)
![](img/decoder.png)

### Motivation

Machine Learning models are often black box, i.e. providing little insight how a prediction is formed. When it comes to the application of ML approaches to scientific domains this problem become crucial since no new research hypothsis is created by a black box model.

This is why we are interested in Machine Learning algorithms which yield a **symbolic expression**.
We want to find equations that map a given set of datapoints.

But the core problem is that the space of symbolic expressions is huge and complex. In this project we look into work that embeds the space of symbolic expression into a dense continous lower-dimensional embedding space that can be easier searched. 

### Description

The model [1] explored in this project relies on a simple auto-encoder structure, i.e. an encoder maps to a latent space and a decoder reconstructs the input.

The steps of the project involve:
* implement an auto-regressive encoder and decoder as a baseline
* constrain latent space by a context-free grammar (as in the paper)
* encoder parses the rules used to "produce" the input string
* sample rules to apply in the decoder
* compare against the baseline

### Evaluation

* Visualize the Latent Space and interpolations in it, e.g.
![](img/interpolation.png)

* Bayesian Optimization for Symbolic Regression in the latent space with RMSE metric

### References

* [1] [Kusner, Matt J., Brooks Paige, and José Miguel Hernández-Lobato. "Grammar variational autoencoder." International Conference on Machine Learning. PMLR, 2017.](https://arxiv.org/abs/1703.01925)

## Data

* Provided in the code source.

## Code

* The authors released the code: https://github.com/mkusner/grammarVAE 