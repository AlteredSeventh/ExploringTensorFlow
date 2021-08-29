import gym

from environment.tetris_gym_environment import TetrisEnv
from tetris_linearschedule import LinearExploration, LinearSchedule
from tetris_nature import NatureQN

from configs import tetris_train_nature as config

if __name__ == '__main__':
    # make env
    env = TetrisEnv(test=True)
    config = config.TestConfig

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
