Perspective Experiments
========================
Jan 2018

Perform various perspective transforms on a 3x3 grid:

(0,2) (1,2) (2,2)
(0,1) (1,1) (2,1)
(0,0) (1,0) (2,0)  = (p_00, p_10, p_20, ...)

The transformed points are

(q_00, q_10, q_20, ...)

The perspective transform is defined by the four target points

(q_01, q_11, q_10, q_00) .

Then look for invariants, like

|q01 - q00| / |q02 - q01|
|q11 - q10| / |q12 - q11|
|q21 - q20| / |q22 - q21|

or

|q10 - q00| / |q11 - q01|
|q11 - q01| / |q12 - q02|

=== The End ===
