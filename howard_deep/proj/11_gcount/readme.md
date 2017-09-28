GCount
=========
AHN, Sep 2017

Take Gcount architecture, but put a dense layer on top to count
B and W stones. See if this will learn to localize stones in the
convolutional feature map. The idea is to learn to find stuff
without having to label it first.

Result:
Works. Batchnorm only in the lower layers, batch_size 4 is better than 16.
No noticeable difference between AveragePooling and Maxpooling.


=== The End ===