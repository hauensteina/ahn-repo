Lambda
=========
AHN, Sep 2017

Classify B and W stones on a 3x3 grid.
Use a purely convolutional network. Tha last layer has
3 channels, corresponding to Black, White, and Empty.
We split the 3x3x3 output into 9 groups of 3, where each
group contains the three channels for one intersection.
Then use softmax on each of the nine intersections to classify.
The splitting was done using Keras.Lambda() before I found out how
to specify the axis on softmax().

This works only up to a 3x3 grid. Try a different network
architecture next time.
