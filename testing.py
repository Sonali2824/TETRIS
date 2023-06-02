import gym
env = gym.make('gym_mdptetris:mdptetris-v2')
print(env.get_state())

env = TetrisFlat(board_height=self.board_height,
                         board_width=self.board_width, seed=self.seed)