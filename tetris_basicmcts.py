
# built on code from @blole blole/ openai-gym-mcts.py
import os
import gym
import sys
import random
import itertools
from time import time
from copy import copy
from math import sqrt, log

from environment.tetris_gym_environment import TetrisEnv

def moving_average(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret

def averaged_value(node):
    if node.visits == 0:
        return 0
    else:
        return node.value / node.visits + sqrt(log(node.parent.visits)/node.visits)

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

# A node in a one-directional graph; a placeholder that will be used to
# keep track of value
class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0

class Simulation:
    def __init__(self, rec_dir, env_name, loops=300, max_depth=1000, playouts=10000):

        if env_name == 'Tetris-v0':
            self.env = TetrisEnv(test=True) #gym.make(self.env_name)
        else:
            raise NotImplementedError

        self.dir = rec_dir+'/'+env_name
        self.loops = loops
        self.max_depth = max_depth
        self.playouts = playouts

    def print_stats(self, loop, score, avg_time):
        sys.stdout.write('\r%3d   score:%10.3f   avg_time:%4.1f s' % (loop, score, avg_time))
        sys.stdout.flush()

    def selection(self, node, state, sum_reward, actions):
        #loop from node down until you reach leaf nodes (i.e. node without children)
        while node.children:
            # step down to a node's unexplored children, one by one until all are explored
            if node.explored_children < len(node.children):
                child = node.children[node.explored_children]
                node.explored_children += 1
                node = child
            else:
            # choose the node with the maximum average value
                node = max(node.children, key=averaged_value)

            # move your state by taking action "node.action" and keep track of this action and its reward
            _, reward, terminal, _ = state.step(node.action)
            sum_reward += reward
            actions.append(node.action)

    def expand(self, node, state ):
        node.children = [Node(node, a) for a in combinations(state.action_space)]
        random.shuffle(node.children)

    def playout(self, state, sum_reward, actions, terminal):
        while not terminal:
            action = state.action_space.sample()
            _, reward, terminal, _ = state.step(action)
            sum_reward += reward
            actions.append(action)

            if len(actions) > self.max_depth:
                sum_reward -= 100
                break

    def update_values(self, node, sum_reward):
        while node:
            node.visits += 1
            node.value += sum_reward
            node = node.parent

    def run(self):
        best_rewards = []
        start_time = time()

        # env.monitor.start(self.dir)
        for loop in range(self.loops):
            self.env.reset()
            root = Node(None, None)

            best_actions = []
            best_reward = float("-inf")

            for _ in range(self.playouts):
                state = copy(self.env)
                # del state._monitor
                sum_reward = 0
                node = root
                terminal = False
                actions = []

                # select best node
                self.selection(node, state, sum_reward, actions)
                # expand best node
                if not terminal: self.expand(node, state)
                # do playout from new nodes
                self.playout(state, sum_reward, actions, terminal )

                # remember best
                if best_reward < sum_reward:
                    best_reward = sum_reward
                    best_actions = actions

                # backpropagate
                self.update_values(node, sum_reward)

                # fix monitors not being garbage collected
                # del state._monitor

            sum_reward = 0
            for action in best_actions:
                _, reward, terminal, _ = self.env.step(action)
                sum_reward += reward
                if terminal:
                    break

            best_rewards.append(sum_reward)
            score = max(moving_average(best_rewards, 100))
            avg_time = (time()-start_time)/(loop+1)
            self.print_stats(loop+1, score, avg_time)
        # env.monitor.close()

def main():
    # get rec_dir
    if not os.path.exists('rec'):
        os.makedirs('rec')
        next_dir = 0
    else:
        next_dir = max([int(f) for f in os.listdir('rec')+["0"] if f.isdigit()])+1
    rec_dir = 'rec/'+str(next_dir)
    os.makedirs(rec_dir)
    print( "rec_dir:", rec_dir )

    # Toy text
    Simulation(rec_dir, 'Tetris-v0', loops=100, playouts=400, max_depth=50).run()
    # Runner(rec_dir, 'NChain-v0', loops=100, playouts=3000, max_depth=50).run()

if __name__ == "__main__":
    main()