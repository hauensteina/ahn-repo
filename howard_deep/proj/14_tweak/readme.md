Tweak
===========
AHN, Oct 2017

Try to get stonecounting and localization to work with up to 19x19.

Results:
9x9
=====
$ 01_generate_imgs.py --resolution 80 --gridsize 9  --ntrain 10000 --nval 1000
$ 02_train.py --gridsize 9 --epochs 10 --rate 0.0001
Architecture 64->128->256->3
- Needs 8 epochs on 10'000 training set 9x9 if empty not labeled
- Needs 9 epochs on 10'000 training set 9x9 if empty labeled

11x11
======
$ 01_generate_imgs.py --resolution 80 --gridsize 11  --ntrain 1000 --nval 100
$ 02_train.py --gridsize 11 --epochs 10 --rate 0.0001

=> Only 25% accuracy after 10 epochs. Bad.
=> Try to increase input res

$ 01_generate_imgs.py --resolution 160 --gridsize 11  --ntrain 1000 --nval 100
=> 44% after 10 epochs
=> 99.95% after 20 epochs
=> 100% after 21 epochs, but total	 overfit (valid at 53)
=> generate a 10'000 image train set
$ 01_generate_imgs.py --resolution 160 --gridsize 11  --ntrain 10000 --nval 1000
$ 02_train.py --gridsize 11 --epochs 10 --rate 0.0001
=> Converges in three epochs without overfitting

13x13
=======
$ 01_generate_imgs.py --resolution 160 --gridsize 13  --ntrain 10000 --nval 1000
$ 02_train.py --gridsize 13 --epochs 10 --rate 0.0001
=> Converges in three epochs without overfitting

15x15
=======
$ 01_generate_imgs.py --resolution 160 --gridsize 15  --ntrain 10000 --nval 1000
$ 02_train.py --gridsize 15 --epochs 10 --rate 0.0001
=> stagnates at 83 pct after 20 epochs
=> inspection shows the input is garbage at resolution 160. Double to 320.
$ 01_generate_imgs.py --resolution 320 --gridsize 15  --ntrain 10000 --nval 1000
=> also stagnates


Experiments to determine Batchnorm and relu vs selu
======================================================
(1) It is important to not do BN or Selu in the highest conv block (last before boiling to 3 channels)
(2) Somtimes selu can replace relu+BN, but not in the lowest layer.
    It is important to BN at least once in the beginning, even with selu
(3) In general, Relu+BN seems more robust than Selu.
    Especially because I need it in the lowest layer anyway.

The (rl BN rl BN rl boil sum) architecture converges in 18 epochs on a 13x13 board, 1000 training samples,
no overfitting. Images generated at 160x160.

Experiments to determine number of filters
============================================
13x13 board, 1000 training samples, images generated at 160x160.

(1) Baseline  (rl BN rl BN rl boil sum) with (32 MP 64 MP (128 64 128) MP (256 128 256) MP)
=> converges in 18 epochs without overfitting

(2) (32 MP 64 MP (128 64 128) MP (128 64 128) MP)
=> does not converge in 25 epochs

(3) (32 MP 64 MP (64 32 64) MP (128 64 128) MP)
=> converges after 25 epochs without overfitting

(4) (32 MP 64 MP (64 16 64) MP (128 64 128) MP)
=> does not converge in 25 epochs

(5) (32 MP 64 MP (32 16 32) MP (64 32 64) MP)
=> does not converge in 25 epochs

(6) (32 MP 64 MP (64 32 64) MP (128 64 128) MP (128 64 128) )
=> does not converge in 25 epochs

(7) (32 MP 64 MP (64 32 64) MP (128 64 128) MP (128 64 128) MP )
=> does not converge in 25 epochs

(8) (16 MP 32 MP (64 32 64) MP (128 64 128) MP)
=> getting close after 25. Converges after 30 without overfitting.

(9) (16 MP 32 MP (64 32 64) MP (128 64 128) MP (256 128 256) MP)
=> does not converge in 30 epochs

(10) Add BatchNorm to layer three in (9)
=> learns the trining set in 17 epochs. Needs exactly 30 to also get validation right.

Result: Stick to the baseline

Last Try: 100 Epochs baseline on 15x15
=======================================
$ 02_train.py --gridsize 15 --epochs 100 --rate 0.0001
=> Converges after 25 epochs without overfitting, on training size set 10000

=== The End ===
