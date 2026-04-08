import pygame
import random
import numpy as np
import pandas as pd

class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines,
        # based on https://github.com/jakejhansen/minesweeper_solver
        rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3, 'no_progress' : -0.3}, gui=False):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self._init_minefield() # Initialize the minefield with bombs and empty spaces, marked with 'B' and 0 respectively
        self._init_numbered_board()
        self.state, self.state_im = self._init_state() # initializes player view, state value U is Unsolved, B is bomb, 0-8 for number of adjacent bombs
        self.n_clicks = 0
        self.n_progress = 0
        self.n_guesses = 0
        self.n_wins = 0
        self.explosion = False
        self.done = False

        self.rewards = rewards

        self.playerfield = np.ones((self.nrows, self.ncols), dtype='int')*9 # The state the player sees, 9 means unclicked, -1 means bomb, 0-8 means number of adjacent bombs, -2 means explosion

        self.gui = False
        if gui:
            self.gui = gui
            self._init_gui()

    def _init_minefield(self):
        self.minefield = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if self.minefield[row][col] != 'B':
                self.minefield[row][col] = 'B'
                mines -= 1

    def _init_numbered_board(self):
        self.numbered_board = self.minefield.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.minefield[x,y] != 'B':
                    coords.append((x,y))

        for coord in coords:
            self.numbered_board[coord] = self._count_bombs(coord)

    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.n_guesses = 0
        self._init_minefield()
        self._init_numbered_board()
        self.state, self.state_im = self._init_state()
        self.done = False
        
        self.explosion = False
        self.playerfield = np.ones((self.nrows, self.ncols), dtype='int')*9

    def step(self, action_index):
        self.done = False
        coords = self.state[action_index]['coord']

        current_state = self.state_im

        # get neighbors before action
        neighbors = self._get_neighbors(coords)

        self._click(action_index)

        # update state image
        new_state_im = self._get_state_im(self.state)
        self.state_im = new_state_im

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            self.done = True
            self.explosion = True

        elif np.sum(new_state_im==-0.125) == self.n_mines: # if win
            reward = self.rewards['win']
            self.done = True
            self.n_progress += 1
            self.n_wins += 1

        elif np.sum(self.state_im == -0.125) == np.sum(current_state == -0.125):
            reward = self.rewards['no_progress']

        else: # if progress
            if all(t==-0.125 for t in neighbors): # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']
                self.n_guesses += 1

            else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated clicks
        
        if self.gui:    
            self._update_playerfield()
            self._render()

        return self.state_im, reward, self.done


    def _get_neighbors(self, coord):
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(self.minefield[row,col])

        return np.array(neighbors)

    def _count_bombs(self, coord):
        neighbors = self._get_neighbors(coord)
        return np.sum(neighbors=='B')

    def _get_state_im(self, state):
        '''
        Gets the numeric image representation state of the board visible to player.
        This is what will be the input for the DQN.
        '''

        state_im = [t['value'] for t in state]
        # Need to keep a third dimension of size 1 because neural network treats that as number of channels
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im
    
    def _update_playerfield(self):
        state_im = [t['value'] for t in self.state]
        state_im = np.reshape(state_im, (self.nrows, self.ncols)).astype(object)
        
        state_im[state_im=='U'] = 9
        state_im[state_im=='B'] = -2 # this way it paints the "red explosion" image
        self.playerfield = state_im.astype(np.int8)

    def _init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self._get_state_im(state)

        return state, state_im

    def _color_state(self, value):
        if value == -1:
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'

    def draw_state(self, state_im):
        state = state_im * 8.0
        state_df = pd.DataFrame(state.reshape((self.nrows, self.ncols)), dtype=np.int8)

        display(state_df.style.applymap(self._color_state))

    def _click(self, action_index):
        coord = self.state[action_index]['coord']
        value = self.numbered_board[coord]

        # ensure first move is not a bomb
        if (value == 'B') and (self.n_clicks == 0):
            minefield = self.minefield.reshape(1, self.ntiles)
            move = np.random.choice(np.nonzero(minefield!='B')[1])
            coord = self.state[move]['coord']
            value = self.numbered_board[coord]
            self.state[move]['value'] = value
        else:
            # make state equal to numbered_board at given coordinates
            self.state[action_index]['value'] = value

        # reveal all neighbors if value is 0
        if value == 0.0:
            self._reveal_neighbors(coord, clicked_tiles=[])

        self._last_click_coords = coord
        self.n_clicks += 1

    def _reveal_neighbors(self, coord, clicked_tiles):
        processed = clicked_tiles
        state_df = pd.DataFrame(self.state)
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows) and
                    ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row,col))

                    index = state_df.index[state_df['coord'] == (row,col)].tolist()[0]

                    self.state[index]['value'] = self.numbered_board[row, col]

                    # recursion in case neighbors are also 0
                    if self.numbered_board[row, col] == 0.0:
                        self._reveal_neighbors((row, col), clicked_tiles=processed)

    def _init_gui(self):
            # Initialize all PyGame and GUI parameters
            pygame.init()
            pygame.mixer.quit() # Fixes bug with high PyGame CPU usage
            self.tile_rowdim = 32 # pixels per tile along the horizontal
            self.tile_coldim = 32 # pixels per tile along the vertical
            self.game_width = self.ncols * self.tile_coldim
            self.game_height = self.nrows * self.tile_rowdim
            self.ui_height = 32 # Contains text regarding score and move #
            self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height+self.ui_height))
            pygame.display.set_caption('Minesweeper')
            # Load Minesweeper tileset
            self.tilemine = pygame.image.load('img/mine.jpg').convert()
            self.tile0 = pygame.image.load('img/0.jpg').convert()
            self.tile1 = pygame.image.load('img/1.jpg').convert()
            self.tile2 = pygame.image.load('img/2.jpg').convert()
            self.tile3 = pygame.image.load('img/3.jpg').convert()
            self.tile4 = pygame.image.load('img/4.jpg').convert()
            self.tile5 = pygame.image.load('img/5.jpg').convert()
            self.tile6 = pygame.image.load('img/6.jpg').convert()
            self.tile7 = pygame.image.load('img/7.jpg').convert()
            self.tile8 = pygame.image.load('img/8.jpg').convert()
            self.tilehidden = pygame.image.load('img/hidden.jpg').convert()
            self.tileexplode = pygame.image.load('img/explode.jpg').convert()
            self.tile_dict = {-1:self.tilemine,0:self.tile0,1:self.tile1,
                            2:self.tile2,3:self.tile3,4:self.tile4,5:self.tile5,
                            6:self.tile6,7:self.tile7,8:self.tile8,
                            9:self.tilehidden, -2:self.tileexplode}
            # Set font and font color
            self.myfont = pygame.font.SysFont('Segoe UI', 32)
            self.font_color = (255,255,255) # White
            self.victory_color = (8,212,29) # Green
            self.defeat_color = (255,0,0) # Red
            # Create selection surface to show what tile the agent is choosing
            self.selectionSurface = pygame.Surface((self.tile_rowdim, self.tile_coldim))
            self.selectionSurface.set_alpha(128) # Opacity from 255 (opaque) to 0 (transparent)
            self.selectionSurface.fill((245, 245, 66)) # Yellow     

    def _render(self):
        # Update the game display after every agent action
        # Accepts a masked array of Q-values to plot as an overlay on the GUI
        # Update and blit text
        text_score = self.myfont.render('SCORE: ', True, self.font_color)
        # text_score_number = self.myfont.render(str(self.score), True, self.font_color)
        text_move = self.myfont.render('MOVE: ', True, self.font_color)
        # text_move_number = self.myfont.render(str(self.move_num), True, self.font_color)
        text_victory = self.myfont.render('VICTORY!', True, self.victory_color)
        text_defeat =  self.myfont.render('DEFEAT!', True, self.defeat_color)         
        self.gameDisplay.fill(pygame.Color('black')) # Clear screen
        self.gameDisplay.blit(text_move, (45, self.game_height+5))
        # self.gameDisplay.blit(text_move_number, (140, self.game_height+5))
        self.gameDisplay.blit(text_score, (400, self.game_height+5))
        # self.gameDisplay.blit(text_score_number, (500, self.game_height+5))
        if self.done:
            if self.explosion:
                self.gameDisplay.blit(text_defeat, (700, self.game_height+5))
            else:
                self.gameDisplay.blit(text_victory, (700, self.game_height+5))
        # Blit updated view of minefield
        self._plot_playerfield()
        """
        if valid_qvalues.size > 0:
            # Blit surface showing agent selection and Q-value representations
            self.selection_animation(np.argmax(valid_qvalues))
            self.plot_qvals(valid_qvalues) """
        pygame.display.update()

    def plot_qvalues_and_next_action(self, action_index, valid_qvalues):
        (x,y) = self.state[action_index]['coord']
        # Superimposes a colored circle over each unrevealed tile in the grid
        # A large blue circle is a tile the agent feels confident is safe
        # A large red circle is a tile the agent feels confident is a mine
        # A small dark/black colored circle is a tile the agent is unsure of
        max_qval = np.max(valid_qvalues)
        min_qval = np.min(valid_qvalues)
        qval_array = valid_qvalues.reshape(self.nrows, self.ncols)
        for k in range(0,self.nrows):
            for h in range(0,self.ncols):
                qval = qval_array[k,h]
                if qval >= 0: # Color blue
                    qval_scale = np.abs((qval / max_qval) ** 0.5)
                    rgb_tuple = (0, 0, int(qval_scale*255))
                else: # Color red
                    qval_scale = np.abs((qval / min_qval) ** 0.5)
                    rgb_tuple = (int(qval_scale*255), 0, 0)
                if (k, h) == (x, y): # Color yellow for the next action
                    rgb_tuple = (255, 255, 0)
                center =  (int(h*self.tile_coldim + self.tile_coldim/2), \
                            int(k*self.tile_rowdim + self.tile_rowdim/2))
                radius = int(self.tile_rowdim/6 * qval_scale)
                pygame.draw.circle(self.gameDisplay, rgb_tuple, center, radius)
        pygame.display.update()     

    def _plot_playerfield(self):
        # Blits the current state's tiles onto the game display
        for k in range(0,self.nrows):
            for h in range(0,self.ncols):
                self.gameDisplay.blit(self.tile_dict[self.playerfield[k,h]], (h*self.tile_coldim, k*self.tile_rowdim))
        if self.n_clicks > 0:
            (x,y) = self._last_click_coords
            tinted_image = self.tile_dict[self.playerfield[x,y]].copy()
            tinted_image.fill((255, 255, 0), special_flags=pygame.BLEND_RGBA_MULT) # Yellow tint
            self.gameDisplay.blit(tinted_image, (y*self.tile_coldim, x*self.tile_rowdim))