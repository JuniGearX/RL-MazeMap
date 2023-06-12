import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 21

HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1000

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
WALL_N = 4

d = {
    1: (255, 175, 0),  # Player color
    2: (0, 255, 0),  # Food color
    3: (0, 0, 255),  # Enemy color
    4: (224, 224, 224),  # Wall color
}

class Blob:
    def __init__(self, spawn_location=None):
        if spawn_location is None:
            self.x = np.random.randint(0, SIZE)
            self.y = np.random.randint(0, SIZE)
        else:
            self.x, self.y = spawn_location

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0)  # Move right
        elif choice == 1:
            self.move(x=-1, y=0)  # Move left
        elif choice == 2:
            self.move(x=0, y=1)  # Move down
        elif choice == 3:
            self.move(x=0, y=-1)  # Move up

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_q_table is None:
    # Initialize the q-table
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

def random_maze(width, height):
    maze = [[True for _ in range(width)] for _ in range(height)]
    for i in range(width):
        for j in range(height):
            if random.random() < 0.25:
                maze[i][j] = False
    return maze

def detect_wall(x, y, maze):
    if x < 0 or x >= SIZE or y < 0 or y >= SIZE or not maze[x][y]:
        return True
    return False

episode_rewards = []

maze = random_maze(SIZE, SIZE)

for episode in range(HM_EPISODES):
    player = Blob()
    while detect_wall(player.x, player.y, maze):
        player = Blob()
    food = Blob()
    while detect_wall(food.x, food.y, maze) or (food.x == player.x and food.y == player.y):
        food = Blob()
    enemy = Blob()
    while detect_wall(enemy.x, enemy.y, maze) or (enemy.x == player.x and enemy.y == player.y) or (enemy.x == food.x and enemy.y == food.y):
        enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif detect_wall(player.x, player.y, maze):
            reward = -MOVE_PENALTY
            player.move(x=-player.x, y=-player.y)  # Move back to the previous position
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # Starts an RGB array of our size
            env[food.x][food.y] = d[FOOD_N]  # Sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # Sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # Sets the enemy location to red
            for i in range(SIZE):
                for j in range(SIZE):
                    if not maze[i][j]:
                        env[i][j] = d[WALL_N]  # Sets the wall location to gray
            img = Image.fromarray(env, 'RGB')  # Reads as RGB
            img = img.resize((400, 400), resample=Image.BOX)  # Resizes the image for better visualization
            cv2.imshow("image", np.array(img))  # Shows the image
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("Episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
