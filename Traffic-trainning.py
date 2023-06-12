import numpy as np
import random
import matplotlib.pyplot as plt

class TrafficGridworldEnv:

    def __init__(self, width, height, num_cars, num_steps):
        self.width = width
        self.height = height
        self.num_cars = num_cars
        self.num_steps = num_steps

        self.grid = np.zeros((self.width, self.height))
        self.cars = []
        for i in range(self.num_cars):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid[x, y] = 1
            self.cars.append((x, y))

        # Create the intersection.
        self.intersection = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # Create the roundabout.
        self.roundabout = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]

        # Create the 2+1 road.
        self.two_plus_one_road = [(5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1)]

        # Create the highway.
        self.highway = [(8, 0), (8, 1), (9, 0), (9, 1)]

        # Set the current state and car.
        self.current_state = (0, 0)
        self.current_car = 0

    def step(self, action):
        # Move the car in the specified direction.
        if action == 0:
            self.current_car = (self.current_car + 1) % self.num_cars
        elif action == 1:
            self.current_car = (self.current_car - 1 + self.num_cars) % self.num_cars
        elif action == 2:
            self.current_state = (self.current_state[0] + 1, self.current_state[1])
        elif action == 3:
            self.current_state = (self.current_state[0] - 1, self.current_state[1])
        elif action == 4:
            self.current_state = (self.current_state[0], self.current_state[1] + 1)
        elif action == 5:
            self.current_state = (self.current_state[0], self.current_state[1] - 1)

        # Check for collisions.
        for i in range(self.num_cars):
            if i != self.current_car and self.current_state == self.cars[i]:
                return -1, None

        # Check for goal state.
        if self.current_state in self.intersection:
            return 10, None
        elif self.current_state in self.roundabout:
            return 20, None
        elif self.current_state in self.two_plus_one_road:
            return 30, None
        elif self.current_state in self.highway:
            return 40, None

        # Otherwise, return a reward of 0.
        return 0, None

    def reset(self):
        self.grid = np.zeros((self.width, self.height))
        self.cars = []
        for i in range(self.num_cars):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid[x, y] = 1
            self.cars.append((x, y))

        self.current_state = (0, 0)
        self.current_car = 0

    def render(self):
        for i in range(self.width):
            for j in range(self.height):
                if self.grid[i, j] == 1:
                    print("C", end=" ")
                else:
                    print(".", end=" ")
            print()

class Vehicle:
    def __init__(self, length, speed, direction):
        self.length = length
        self.speed = speed
        self.direction = direction

    def move(self, grid):
        # Check if the vehicle can move in the specified direction
        if not self.can_move(grid):
            return

        # Move the vehicle in the specified direction
        self.position = self.position + self.direction * self.speed

    def can_move(self, grid):
        # Check if the vehicle is colliding with another vehicle
        for other_vehicle in grid.vehicles:
            if self.collides_with(other_vehicle):
                return False

        # Check if the vehicle is outside of the grid
        if not (0 <= self.position[0] < grid.width and 0 <= self.position[1] < grid.height):
            return False

        return True

    def collides_with(self, other_vehicle):
        # Check if the two vehicles are overlapping
        return self.position[0] == other_vehicle.position[0] and self.position[1] == other_vehicle.position[1]

def generate_vehicles(grid):
    # Generate a random number of vehicles
    num_vehicles = random.randint(1, 10)

    # Create a list of vehicles
    vehicles = []
    for i in range(num_vehicles):
        # Choose a random type of vehicle
        type = random.choice(["truck", "car", "bus"])

        # Choose a random position for the vehicle
        x = random.randint(0, grid.width - 1)
        y = random.randint(0, grid.height - 1)
        position = (x, y)

        # Choose a random speed for the vehicle
        speed = random.randint(1, 10)

        # Choose a random direction for the vehicle
        direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])

        # Create a new vehicle
        vehicle = Vehicle(type, speed, direction)
        vehicle.position = position

        # Add the vehicle to the list of vehicles
        vehicles.append(vehicle)

    return vehicles

# Define the state space
num_states = 10  # Assuming 10 possible states based on the given code

# Define the action space
num_actions = 6  # 6 possible actions based on the given code

# Initialize the Q-table with zeros
q_table = np.zeros((num_states, num_actions))

# Define the reward system
rewards = {
    -1: -10,  # Collision: -10
    10: 100,  # Intersection: 100
    20: 200,  # Roundabout: 200
    30: 300,  # 2+1 road: 300
    40: 400   # Highway: 400
}

def run_traffic_simulation(TrafficGridworldEnv, q_table):
    rewards = []
    success_rates = []
    num_episodes = 1000  # Number of simulation episodes

    for episode in range(num_episodes):
        total_reward = 0
        success_count = 0

        TrafficGridworldEnv.reset(self)
        done = False

        while not done:
            state = get_state(TrafficGridworldEnv)
            action = np.argmax(q_table[state])
            reward, done = TrafficGridworldEnv.step(action)
            total_reward += reward

            if reward > 0:
                success_count += 1

        rewards.append(total_reward)
        success_rates.append(success_count / TrafficGridworldEnv.num_cars)

    return rewards, success_rates

def get_state(TrafficGridworldEnv):
    state = 0
    if TrafficGridworldEnv.current_state in TrafficGridworldEnv.intersection:
        state += 1
    if TrafficGridworldEnv.current_state in TrafficGridworldEnv.roundabout:
        state += 2
    if TrafficGridworldEnv.current_state in TrafficGridworldEnv.two_plus_one_road:
        state += 3
    if TrafficGridworldEnv.current_state in TrafficGridworldEnv.highway:
        state += 4
    state += TrafficGridworldEnv.current_car * 10
    return state

# Assuming you have instantiated the TrafficGridworldTrafficGridworldEnv as 'TrafficGridworldEnv' and trained the Q-table as 'q_table'

# Run the traffic simulation and get the rewards and success rates
rewards, success_rates = run_traffic_simulation(TrafficGridworldEnv, q_table)

# Plot the reward accumulation
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward Accumulation')
plt.show()

# Plot the success rate
plt.plot(success_rates)
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('Success Rate')
plt.show()
