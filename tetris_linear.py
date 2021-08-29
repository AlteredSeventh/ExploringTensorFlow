import tensorflow as tf
import tensorflow.contrib.layers as layers

from environment.tetris_gym_environment import TetrisEnv
from core.deep_q_learning import DQN
from tetris_linearschedule import LinearExploration, LinearSchedule


from configs import config
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)

        board_height, board_width = state_shape[0], state_shape[1]
        self.s = tf.placeholder(dtype=tf.uint8,
                                shape=[None, board_height, board_width,  self.config.state_history],
                                name='state')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None], name='action')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.sp = tf.placeholder(dtype=tf.uint8,
                                 shape=[None, board_height, board_width,  self.config.state_history],
                                 name='next_state')
        self.done_mask = tf.placeholder(dtype=tf.bool, shape=[None], name='done_mask')
        self.lr = tf.placeholder(dtype=tf.float32, shape=(), name='lr')

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, board height, board width)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        state_flatten = layers.flatten(state, scope=scope)
        out = layers.fully_connected(state_flatten, num_actions, reuse=reuse, scope=scope, activation_fn=None)

        return out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        In DQN, we maintain two identical Q networks with
        2 different sets of weights.
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        q_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        op = [tf.assign(target_q_collection[i], q_collection[i]) for i in range(len(q_collection))]
        self.update_target_op = tf.group(*op)

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        """
          The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        """
        Q_samp = self.r + (1 - tf.cast(self.done_mask, tf.float32)) * \
                 self.config.gamma * tf.reduce_max(target_q, axis=1)
        Q = tf.reduce_sum(q * tf.one_hot(self.a, num_actions), axis=1)
        self.loss = tf.reduce_mean((Q_samp - Q) ** 2)

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store
                this scalar in self.grad_norm
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        scope_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.loss, scope_variable)
        if self.config.grad_clip:
            clipped_grads_and_vars = [(tf.clip_by_norm(item[0], self.config.clip_val), item[1]) for item in
                                      grads_and_vars]
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)
        self.grad_norm = tf.global_norm([item[0] for item in grads_and_vars])


if __name__ == '__main__':
    env = TetrisEnv(test = True)
    config = config.TestConfig

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
