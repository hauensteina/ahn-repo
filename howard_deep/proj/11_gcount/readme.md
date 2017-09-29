GCount
=========
AHN, Sep 2017

Take Gcount architecture, use a Lambda layer to sum and get number of
B and W stones. See if this will learn to localize stones in the
convolutional feature map. The idea is to learn to find stuff
without having to label it first.

Result:
Works. Batchnorm only in the lower layers, batch_size 4 is better than 16.
No noticeable difference between AveragePooling and Maxpooling.
Replacing pooling with a larger stride breaks convergence.

=== The End ===
