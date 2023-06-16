# Code adapted from: Andrean Lay. 2020. Tetris AI Deep Q-Learning. [Code]. GitHub. Available at: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2 as cv
import random
from typing import Optional, Union

#Testing: Tetrimino Ordering
tetrimino_ordering = ["I", "I", "T", "S", "O", "L", "L", "T", "O", "L", "O", "Z", "S", "Z", "T", "S", "T", "J", "I", "I", "I", "I"]
# Tetriminoes
tetromino_colors = {
     "cyan" : (0, 255, 255),   
     "blue" : (0, 0, 255),    
     "pink" : (255, 51, 153),  
     "yellow" : (255, 255, 0),   
     "green" : (0, 255, 0),     
     "purple" : (128, 0, 128),   
     "red" : (255, 0, 0),      
     "black": (0, 0, 0)
}

tetromino_color_names= {
     0 : "cyan", 
     1 : "blue",     
     2 : "pink",   
     3 : "yellow",   
     4 : "green",     
     5 : "purple",   
     6 : "red"      
}
shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'S': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'Z': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
} # interchanged S and Z
shape_names = ['T', 'J', 'L', 'S', 'Z', 'I', 'O'] # interchanged S and Z

# shapes = { 'J': [(0, 0), (-1, 0), (0, -1), (0, -2)]}
# shape_names = ['J']
green = (156, 204, 101)
black = (0, 0, 0)
white = (255, 255, 255)


# Helper Functions
def rotated(shape):
    return [(-j, i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False

def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)

def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new

class Tetris(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=float)
        #self.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
        #self.board = np.array(self.board)
        self.board_colors = np.full((width, height), "black", dtype="U10")
        # State size (Cleared lines, bumpiness, holes, height, landing height, row transitions, column transitions, next piece)
        self.state_size = 8

        # For running the engine
        self.score = -1
        self.anchor = None
        self.shape = None
        self.invalid = False
        self.selected_tetrimino = 0
        self.tetrimino_board_pos = []
        self.color = "black"

        # Testing
        self.tetrimino_ordering_index = 0

        # Used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # Reset after initialising
        self.reset()

        # Defining action space

        self.num_rotations = 4  # Number of possible rotations

        # Define the action space
        self.action_space = gym.spaces.discrete.Discrete(self.width * self.num_rotations)


        # Defining observation space
        self.observation_space = spaces.Box(low=0, high=50000, shape=(self.state_size,), dtype=float)

        

   
    # Helper Functions
    def map_discrete_to_tuple(self, discrete_action):
        discrete_action = int(discrete_action)
        actions = []
        for i in range(self.width):
            for j in range(self.num_rotations):
                actions.append((i, j))
        action_dict = {}
        index = 0

        for i in actions:
            action_dict[index] = i
            index += 1
       
        return action_dict[discrete_action]

    def _choose_shape(self):

        max_count = max(self._shape_counts)

        tetromino = None
        valid_tetrominos = [shape_names[i] for i in range(len(shapes)) if self._shape_counts[i] < max_count]
        if len(valid_tetrominos) == 0:
            tetromino = random.sample(shape_names, 1)[0]
        else:
            tetromino = random.sample(valid_tetrominos, 1)[0]
        
        # Testing 
        tetromino = tetrimino_ordering[self.tetrimino_ordering_index]
        if self.tetrimino_ordering_index == len(tetrimino_ordering)-1:
            self.tetrimino_ordering_index = 0
        else:
            self.tetrimino_ordering_index += 1
        
        # End of Testing code

        self._shape_counts[shape_names.index(tetromino)] += 1
        self.selected_tetrimino = shape_names.index(tetromino)
        self.color = tetromino_color_names[self.selected_tetrimino]
        #print("shape", tetromino)
        return shapes[tetromino] 

    def _new_piece(self):
        self.anchor = (self.width / 2, 1)
        self.shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        new_board_color = np.full((self.width, self.height), "black", dtype="U10")
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                new_board_color[:, j] = self.board_colors[:, i]
                j -= 1
        self.score += sum(can_clear)    
        self.board = new_board
        self.board_colors = new_board_color

        return sum(can_clear)

    def valid_action_count(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum
    
    # Step
    def step(self, action):
        action = self.map_discrete_to_tuple(action)
        # Set current state wrt current piece
        current_states = self.get_next_states()
        current_state =  current_states[action]

        invalid_actions = [key for key, value in current_states.items() if all(elem == 0 for elem in value)]
        if action in invalid_actions:
            self.invalid = True

        pos = [action[0], 0]

        # Rotate shape n times
        for rot in range(action[1]):
            self.shape = rotated(self.shape)

        self.shape, self.anchor = hard_drop(self.shape, pos, self.board)

        reward = 0
        done = False

        self._set_piece(True)

        cleared_lines = self._clear_lines()
        reward += cleared_lines ** 2 * self.width + 1

        if self.invalid:
            reward -= 5
            self.invalid = False

        if np.any(self.board[:, 0]):
            self.reset()
            done = True
            reward -= 5
        else:
            self._new_piece()

        # Obtain next states on taking action wrt next piece
        next_states = self.get_next_states()

        info = {"current_piece_states": current_states, "next_piece_states": next_states}
        return current_state, reward, done, False, info
    
    def reset(self, seed:Optional[int] = None):
        super().reset(seed=seed)
        self.time = 0
        self.score = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)
        #self.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
        #self.board = np.array(self.board)
        self.board_colors =  np.full((self.width, self.height), "black", dtype="U10")
        return np.array([0 for _ in range(self.state_size)]), {"next_piece_states": self.get_next_states()}

    def _set_piece(self, on):
        """To lock a piece in the board"""
        tetrimino_board_pos = []
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on
                tetrimino_board_pos.append((x, y))
                # Coloring
                if on:
                    self.board_colors[int(self.anchor[0] + i), int(self.anchor[1] + j)] = self.color
                else:
                    self.board_colors[int(self.anchor[0] + i), int(self.anchor[1] + j)] = "black"
        
        self.tetrimino_board_pos = tetrimino_board_pos
                

    def _clear_line_dqn(self, board, board_color):
        can_clear = [np.all(board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(board)
        new_board_color =  np.full((self.width, self.height), "black", dtype="U10")
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                new_board_color[:, j] = self.board_colors[:, i]
                j -= 1
        self.score += sum(can_clear)    
        board = new_board
        board_color = new_board_color

        return sum(can_clear), board, board_color

    def get_bumpiness_height(self, board):
        bumpiness = 0
        columns_height = [0 for _ in range(self.width)]

        for i in range(self.width): 
            for j in range(self.height):
                if board.T[j][i]:
                    columns_height[i] = self.height - j
                    break
        for i in range(1, len(columns_height)):
            bumpiness += abs(columns_height[i] - columns_height[i-1])

        return bumpiness, sum(columns_height)

    def get_landing_height(self, board, x):
        # Find the lowest point of the Tetrimino

        if (self.tetrimino_board_pos):
            max_value = max(coord[1] for coord in self.tetrimino_board_pos)
        else:
            max_value = self.width - 1
        

        # Calculate the landing height
        landing_height = self.height - max_value - 1
        
        return landing_height

    
    def get_row_transitions(self, board):
        num_transitions = 0
        for i in range(self.height): 
            hasTransition = False
            for j in range(self.width-1): #0-8
                if board[j][i]!=board[j+1][i]:
                    hasTransition = True
                    num_transitions+=1
                
            if (not board[0][i]): # and hasTransition:
                num_transitions+=1
            if (not board[self.width-1][i]): # and hasTransition:
                num_transitions+=1
        return num_transitions
    
    def get_column_transitions(self, board):
        num_transitions = 0
        for i in range(self.width): # 0-9 
            hasTransition = False
            for j in range(self.height-1): # 0-18
                if board.T[j][i]!=board.T[j+1][i]:
                    hasTransition = True
                    num_transitions+=1
                
            if (board.T[0][i]): # and hasTransition:
                num_transitions+=1
            if (not board.T[self.height-1][i]): # and hasTransition:
                num_transitions+=1
        return num_transitions
    
    def get_holes(self, board):
        holes = 0

        for col in zip(*board.T):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            holes += len([x for x in col[row + 1:] if x == 0])

        return holes

    def get_current_state(self, board, landing_height):
        # Getting lines which can be cleared and the new cleared board
        cleared_lines, board, board_color = self._clear_line_dqn(board, self.board_colors)

        # Getting number of holes that are impossible to fill
        holes = self.get_holes(board)

        # Getting bumpiness / sum of difference between each adjacent column
        bumpiness, height = self.get_bumpiness_height(board)

        # Row Transitions
        row_transitions = self.get_row_transitions(board)

        # Column Transitions
        column_transitions = self.get_column_transitions(board)

        return np.array([cleared_lines, holes, bumpiness, height, landing_height, row_transitions, column_transitions, self.selected_tetrimino]) 


    def get_next_states(self):
        """To get all possible state from current shape"""
        old_shape = self.shape
        old_anchor = self.anchor
        states = {}
        # Loop to try each posibilities
        for rotation in range(4):
            max_x = int(max([s[0] for s in self.shape]))
            min_x = int(min([s[0] for s in self.shape]))

            for x in range(abs(min_x), self.width - max_x):
                # Try current position
                pos = [x, 0]
                while not is_occupied(self.shape, pos, self.board):
                    pos[1] += 1
                pos[1] -= 1

                self.anchor = pos
                self._set_piece(True)
                landing_height = self.get_landing_height(self.board, x)
                states[(x, rotation)] = self.get_current_state(self.board[:], landing_height)
                self._set_piece(False)
                self.anchor = old_anchor

            self.shape = rotated(self.shape)
        for i in range(self.width):
            for j in range(4):
                if (i,j) not in states:
                    states[(i, j)] = np.array([0 for _ in range(self.state_size)])
        return states

    def render(self):
        self._set_piece(True)
        board = self.board_colors[:].T

        board = [[tetromino_colors[board[i][j]]  for j in range(self.width)] for i in range(self.height)]
        self._set_piece(False)

        img = np.array(board).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = cv.resize(img, (self.width * 25, self.height * 25), interpolation=cv.INTER_NEAREST)

        # To draw lines every 25 pixels
        img[[i * 25 for i in range(self.height)], :, :] = 0
        img[:, [i * 25 for i in range(self.width)], :] = 0

        # Add extra spaces on the top to display game score
        extra_spaces = np.zeros((2 * 25, self.width * 25, 3))
        #cv.putText(extra_spaces, "Score: " + str(score), (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1, white, 2, cv.LINE_AA)

        # Add extra spaces to the board image
        img = np.concatenate((extra_spaces, img), axis=0)

        # Draw horizontal line to separate board and extra space area
        img[50, :, :] = white

        cv.imshow('DQN Tetris', img)
        cv.waitKey(1)

   

        
