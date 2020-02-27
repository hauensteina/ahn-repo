import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from qlearn import Catch, Action

#--------------
def main():
    # Make sure this grid size matches the value used for training
    grid_size = 10

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(10):
        loss = 0.
        env.reset()
        # env.state.fruit_col = 0 # Testing edge case
        game_over = False
        # Get initial input. This is the canvas, linearized into a list.
        next_input = env.observe()

        plt.imshow( next_input.reshape((grid_size,)*2), interpolation='none', cmap='gray')
        plt.savefig("%03d.png" % c)
        c += 1
        while not game_over:
            cur_input = next_input
            # get next action
            q = model.predict( cur_input[np.newaxis])[0]
            action = Action.index( np.argmax( q))

            # apply action, get rewards and new state
            next_input, reward, game_over = env.act( action)

            plt.imshow( next_input.reshape((grid_size,)*2), interpolation='none', cmap='gray')
            plt.savefig("%03d.png" % c)
            c += 1


if __name__ == "__main__":
    main()
