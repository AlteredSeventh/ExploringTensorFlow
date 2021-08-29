import tensorflow as tf
import tensorflow.contrib.layers as layers

from environment.tetris_gym_environment import TetrisEnv
from tetris_linearschedule import LinearExploration, LinearSchedule
from tetris_linear import Linear
from configs import tetris_train_nature as config

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        #   todo: config
        # architecture based on:
        # https: // pdfs.semanticscholar.org / d9de / 866d4c4effe9ad74b20e0e4daa11eddbcc3b.pdf?_ga = 2.104128000.1270170755.1585654811 - 90523069.1585654811

        with tf.variable_scope(scope, reuse=reuse) as _:
            # out = layers.conv2d(out, num_outputs=32, kernel_size=3)
            # out = layers.dropout(out, 0.75)
            # out = layers.conv2d(out, num_outputs=32, kernel_size=3)
            # out = layers.dropout(out, 0.75)
            # out = layers.conv2d(out, num_outputs=64, kernel_size=3)
            # out = layers.dropout(out , 0.75)
            # out = layers.conv2d(out, num_outputs=128, kernel_size=3)
            # out = layers.dropout(out, 0.75)
            # out = layers.conv2d(out, num_outputs=128, kernel_size=1)
            # out = layers.dropout(out, 0.75)
            # out = layers.conv2d(out, num_outputs=128, kernel_size=3)
            # out = layers.flatten(out)
            # out = layers.fully_connected(out, num_outputs=128)
            # out = layers.fully_connected(out, num_outputs=512)
            # out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            #
            # 1 original Mnih Nature paper
            # out = layers.conv2d(out, num_outputs=32, kernel_size=8, stride=4)
            # out = layers.conv2d(out, num_outputs=64, kernel_size=4, stride=2)
            # out = layers.conv2d(out, num_outputs=64, kernel_size=3, stride=1)
            # out = layers.flatten(out)
            # out = layers.fully_connected(out, num_outputs=512)
            # out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            # JW special
            out = layers.conv2d(out, num_outputs=32, kernel_size=4, stride=4)
            out = layers.conv2d(out, num_outputs=64, kernel_size=3, stride=2)
            out = layers.conv2d(out, num_outputs=64, kernel_size=2, stride=1)
            out = layers.flatten(out)
            out = layers.fully_connected(out, num_outputs=512)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    # make env
    env = TetrisEnv(test=True)
    config = config.TestConfig


    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
