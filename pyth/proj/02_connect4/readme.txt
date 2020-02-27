
Readme for connect4 with reinforcement learning like AlphaZero
=================================================================
Original from
https://github.com/plkmo/AlphaZero_Connect4/blob/master/src/connect_board.py

AHN, Feb 21, 2020

How to play against the trained net
-------------------------------------

$ python play_against_c4.py

This will only work on a box with a GPU, e.g marfa.

The network is in
model_data/c4_current_net_trained_iter8.pth.tar

It is not in this repo. Get it from the plkmo repo above.

How to train a net
---------------------
$ time python main_pipeline.py --MCTS_num_processes 2 --num_games_per_MCTS_process 2 --num_evaluator_games 1 --num_epochs 10 --total_iterations 3

real    33m4.319s
user    56m0.228s
sys     1m18.208s

The pipeline is:

generate MCTS self play training games -> train -> match between new and prev strongest

This flow is called one iteration.


Explanation of parameters:

--MCTS_num_processes 2 --num_games_per_MCTS_process 2:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each of two processes will generate two training games by MCTS self play.
So in the above example, one iteration will use four training games.
The default is 5 proc * 120 games = 6000 games.
Game generation and game playing is abysmally slow.
Needs to be at least 10 times faster.

--num_epochs 10
~~~~~~~~~~~~~~~~~
Training epochs after generating the MCTS games.
The default is 300.
Much faster than game generation.

--num_evaluator_games 1
~~~~~~~~~~~~~~~~~~~~~~~~~~
How many gmaes to play to determine whether the new net is stronger.
Leela uses 400 for this. The default is 100.

--total_iterations 3
~~~~~~~~~~~~~~~~~~~~~~
How often to go through the complete cycle.
Default is 1000. Looks like you just let this run until you get tired of it
or you die of old age.


=== The End ===
