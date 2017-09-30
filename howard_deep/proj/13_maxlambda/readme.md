MaxLambda
===========
AHN, Sep 2017

Get rid of the empty count. Each pixel is W,B,or empty,
but only the wihite and black counts are trained.

Get resolution down to 20x20, but not lower.

Instead of maxpooling, replace each element with its softmax in the 3x3
neighborhood. An attempt to have the benefits of maxpooling without losing
positinal info.


=== The End ===
