Googlenet
=========
AHN, Sep 2017

Classify B and W stones on a nxn grid.
Use a purely convolutional network inspired by the YOLO9000
paper, which in turn says it's inspired by Googlenet and
NiN (Network in Network).
Hopefully this will work on larger grids than 3x3.


Result:

This was hard bcecause matrix rows go from top to bottom,
but image y coords go from bottom to top.
Reversing the matrix rows in the targets made this an easy problem.

=== The End ===
