CopyConv
===========
AHN, Sep 2017

Same as 11_gcount, but initialize initial model by reading parameters
for the convolutional layers from init_weights.h5.
If a model.h5 is found, it will overwrite those weights.

The idea is to either initialize with good convolutional weights,
or even to just reuse them for a diferent gridsize/resolution.

Convolutional weights are always saved to conv_weights.h5.
To use them for initialization, move them to init_weights.h5.

=== The End ===
