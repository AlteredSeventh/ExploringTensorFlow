import numpy as np

class Config():
    board_width  = 100
    board_height = 400
    shape_max_width  = 10
    numberOf90DegRotations = 3
    score_method = "basicCoverageScore"
    stopping_condition_score_diff = 0.001
    stopping_condition_step_diff = 10

   # env config
    render_train     = False
    render_test      = False
    env_name = "Tetris-v0"
    overwrite_render = True
    record           = False
    high             = 400

    # output config
    output_path  = "results/tetris_linear/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 50
    grad_clip = True
    clip_val = 10
    saving_freq = 250000
    log_freq = 50
    eval_freq = 250000
    record_freq = 250000
    soft_epsilon = 0.05

    # hyper params
    nsteps_train       = 10000
    batch_size         = 32
    buffer_size        = 1000
    target_update_freq = 500
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 0
    lr_begin           = 0.005
    lr_end             = 0.001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = nsteps_train/2
    learning_start     = 200

class TestConfig():
    board_width = 12
    board_height = 40
    shape_max_width = 4
    numberOf90DegRotations = 3
    score_method = "basicCoverageScore"
    stopping_condition_score_diff = 0.0
    stopping_condition_step_diff = 20

   # env config
    env_name = "Tetris-Test-v0"
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = True
    high             = 400

    # output config
    output_path  = "results/tetris_linear/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip = True
    clip_val = 10
    saving_freq = 250000
    log_freq = 50
    eval_freq = 250000
    record_freq = 250000
    soft_epsilon = 0.05

    # hyper params
    nsteps_train = 500000
    batch_size = 32
    buffer_size = 1000000
    target_update_freq = 1000
    gamma = 0.99
    learning_freq = 4
    state_history = 1
    skip_frame = 4
    lr_begin = 0.00025
    lr_end = 0.00005
    lr_nsteps = nsteps_train / 2
    eps_begin = 1
    eps_end = 0.1
    eps_nsteps = 1000000
    learning_start = 50000

    zeroInitializedBoard = np.zeros((board_height, board_width))
