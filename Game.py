import numpy as np
from PIL import Image
import cv2
from random import randint

class Snake:
    def __init__(self, x, y):
        self.x = randint(0, x-1)
        self.y = randint(0, y-1)
        self.body = [[self.x,self.y]]
        self.tail_prev_pos = self.body[-1].copy()

    def move(self, x, y):
        self.tail_prev_pos = self.body[-1].copy()

        # move body
        new_body = self.body.copy()
        for i in range(len(self.body)-1):
            new_body[i+1] = self.body[i].copy()
        self.body = new_body

        # move head
        self.body[0][0] += x
        self.body[0][1] += y

        self.x = self.body[0][0]
        self.y = self.body[0][1]

    def grow(self):
        self.body.append(self.tail_prev_pos)

class Game:
    def __init__(self):
        self.dimension = 10
        self.snake = Snake(self.dimension, self.dimension)
        self.food = [None, None]
        self.num_actions = 4
        self.food_reward = 1
        self.death_reward = -1
        self.place_food()
        self.done = False

    def place_food(self):
        food_x = randint(0, self.dimension-1)
        food_y = randint(0, self.dimension-1)
        while [food_x, food_y] in self.snake.body:
            food_x = randint(0, self.dimension-1)
            food_y = randint(0, self.dimension-1)
        self.food = [food_x, food_y]

    def reset(self):
        self.done = False
        self.snake = Snake(self.dimension, self.dimension)
        self.place_food()
        return self.get_state(), 0

    def get_state(self):
        # state is an array the size of the game board
        # empty spaces are -1's, snake body is 0.5, snake head is 1 and food is -0.5
        # these values where chose as if to represent pixel brightness in a greyscale image
        state =  np.full((self.dimension, self.dimension), -1, dtype=np.float32)
        state[self.food[0]][self.food[1]] = -0.5
        # draw head only if not done
        if not self.done:
            state[self.snake.body[0][0]][self.snake.body[0][1]] = 1
        for i in range(1, len(self.snake.body)):
            state[self.snake.body[i][0]][self.snake.body[i][1]] = 0.5
        return state.reshape(10,10,1)

    def step(self, action):
        reward = 0
        if action == 0: # move right
            self.snake.move(1,0)
        elif action == 1: # move left
            self.snake.move(-1,0)
        elif action == 2: # move up
            self.snake.move(0,1)
        elif action == 3: # move down
            self.snake.move(0,-1)

        if (self.snake.x == self.dimension) or (self.snake.y == self.dimension):
            # snake hit wall (case 1)
            self.done = True
            reward = self.death_reward
        elif (self.snake.x < 0) or (self.snake.y < 0):
            # snake hit wall (case 2)
            self.done = True
            reward = self.death_reward
        elif (self.snake.x == self.food[0]) and (self.snake.y == self.food[1]):
            # snake got food
            self.place_food()
            self.snake.grow()
            reward = self.food_reward
        elif [self.snake.x, self.snake.y] in self.snake.body[1:]:
            # snake hit body
            self.done = True
            reward = self.death_reward

        # return new state
        return self.get_state(), reward, self.done

    def render(self):
        if self.done:
            return
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        board = np.zeros((self.dimension, self.dimension, 3), dtype=np.uint8)
        food_color = (0, 0, 255)
        snake_color = (255, 175, 0)
        snake_head = (175, 175, 0)

        # loop over snake rendering whole body
        board[self.snake.body[0][0]][self.snake.body[0][1]] = snake_head
        for i in range(1, len(self.snake.body)):
            board[self.snake.body[i][0]][self.snake.body[i][1]] = snake_color

        board[self.food[0]][self.food[1]] = food_color

        img = Image.fromarray(board, "RGB")
        img = img.resize((300,300))
        cv2.imshow("Snake", np.array(img))
