import json
import numpy as np
from enum import Enum
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

from pdb import set_trace as BP

#=======================
class Action( Enum):
    left = 0
    stay = 1
    right = 2

    @classmethod
    # A random action
    #----------------------
    def random( cls):
        return list(Action)[np.random.randint(len(Action))]

    @classmethod
    # Action by index
    #----------------------
    def index( cls, idx):
        return list(Action)[idx]

    # left=-1, stay=0, right=1
    #--------------------------
    def direction( self):
        return self.value - 1

#==============
class State:
    def __init__( self, fruit_row, fruit_col, basket_col):
        self.fruit_row = fruit_row
        self.fruit_col = fruit_col
        self.basket_col = basket_col

#======================
class Catch(object):
    #---------------------------------
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    #----------------------------------
    def _update_state(self, action):
        new_basket = self.state.basket_col + action.direction()
        new_basket = min( max( 1, new_basket), self.grid_size-2)
        self.state = State( self.state.fruit_row + 1, self.state.fruit_col, new_basket)

    #-------------------------------------------------------
    def _draw_state(self):
        im_size = (self.grid_size,)*2 # (10,10)
        canvas = np.zeros(im_size)
        canvas[ self.state.fruit_row, self.state.fruit_col] = 1
        canvas[-1, self.state.basket_col-1 : self.state.basket_col+2] = 1
        return canvas

    #-----------------------
    def _get_reward(self):
        if self.state.fruit_row == self.grid_size-1:
            if abs(self.state.fruit_col - self.state.basket_col) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    #---------------------
    def _is_over(self):
        if self.state.fruit_row == self.grid_size-1:
            return True
        else:
            return False

    #--------------------
    def observe(self):
        canvas = self._draw_state()
        # Linearize canvas into a 1D list. Nice trick.
        return canvas.reshape((1, -1))[0]

    #------------------------
    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    #------------------
    def reset(self):
        fruitcol  = np.random.randint( 0, self.grid_size)
        basketcol = np.random.randint( 1, self.grid_size-1)
        fruitrow = 0
        self.state = State( fruitrow, fruitcol, basketcol)

#====================================
class ExperienceReplay(object):
    #----------------------------------------------------
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = []
        self.discount = discount

    # Transition is [cur_input, action, reward, next_input]
    #---------------------------------------------------------
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    #-------------------------------------------
    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        batch_size = min(len_memory, batch_size)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[0]
        inputs = np.zeros( (batch_size, env_dim) )
        targets = np.zeros( (batch_size, num_actions) )
        for i, rrandom in enumerate( np.random.randint(0, len_memory, size=batch_size)):
            cur_input, action, reward, next_input = self.memory[rrandom][0]
            game_over = self.memory[rrandom][1]
            inputs[i] = cur_input
            # Predict wants a list of inputs, we have only one.
            # Newaxis turns our one input into a list with one element.
            # Predict comes back with a list of length 1, and [0] gets rid of the list.
            targets[i] = model.predict( cur_input[np.newaxis])[0] # Estimate reward for each possible action.
            # Now overwrite the reward estimate for the one action we actually took.
            # This is where we learn something because we use the result of our action, called next_input.
            Q_sa = np.max( model.predict( next_input[np.newaxis])[0]) # Estimate best reward from next_input
            if game_over:
                if reward not in [-1,1]:
                    BP()
                    tt=42
                targets[i, action.value] = reward
            else:
                if reward != 0:
                    BP()
                    tt=43
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action.value] = reward + self.discount * Q_sa
        return inputs, targets

#-----------
def main():
    EPSILON = .1  # exploration
    N_EPOCHS = 1000
    MAX_MEMORY = 500
    HIDDEN_SIZE = 100
    BATCH_SIZE = 50
    GRID_SIZE = 10
    MODEL_FNAME = 'model.h5'

    model = Sequential()
    model.add( Dense( HIDDEN_SIZE, input_shape=(GRID_SIZE**2,), activation='relu'))
    model.add( Dense( HIDDEN_SIZE, activation='relu'))
    model.add( Dense( len(Action)))
    model.compile( sgd( lr=.2), "mse")

    try:
        model.load_weights( MODEL_FNAME)
        print( '>>>>> loading weights from ' + MODEL_FNAME)
    except:
        print( '>>>>> starting with random model')

    # Define environment/game
    env = Catch( GRID_SIZE)

    # Initialize experience replay object
    exp_replay = ExperienceReplay( max_memory=MAX_MEMORY)

    # Train
    win_cnt = 0
    for e in range( N_EPOCHS): # Each epoch is one game
        loss = 0.
        env.reset()
        game_over = False
        # Get initial input. This is the canvas, linearized into a list.
        next_input = env.observe()

        # Play one whole game
        while not game_over:
            cur_input = next_input
            # get next action
            if np.random.rand() <= EPSILON:
                action = Action.random()
            else:
                q = model.predict( cur_input[np.newaxis])[0]
                action = Action.index( np.argmax( q))

            # apply action, get reward and new state
            next_input, reward, game_over = env.act( action)
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember( [cur_input, action, reward, next_input], game_over)

            # adapt model after each action
            inputs, targets = exp_replay.get_batch( model, batch_size=BATCH_SIZE)
            loss += model.train_on_batch( inputs, targets)

        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights( MODEL_FNAME, overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

if __name__ == "__main__":
    main()
