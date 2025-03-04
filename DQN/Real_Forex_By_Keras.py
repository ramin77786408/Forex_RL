import numpy as np
import pandas as pd
import random
from gym import spaces
import keras
from keras import layers

from tensorflow.keras.losses import Huber
import tensorflow as tf
import MetaTrader5 as mt5
from ta.volatility import BollingerBands

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
random.seed(42)
# Set option to display all columns
pd.set_option('display.max_columns', None)
# This approach ensures that all columns of the DataFrame are visible when printed in the terminal.
pd.reset_option('display.max_columns')


def save_number(number, filename='number.txt'):
    """Save a number to text file"""
    with open(filename, 'w') as f:
        f.write(str(number))

def load_number(filename='number.txt', default=0.0):
    """Load a number from text file"""
    try:
        with open(filename, 'r') as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return default

# Example usage:
best_reward = load_number('best_reward.txt', default=0.0)


num_epochs=10
point = 100000
symbol = "EURUSD"
episode_reward = 0.0
# Configuration
gamma = 0.99  # Discount factor for past rewards
batch_size = 512  # Size of batch taken from replay buffer
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
# Number of frames to take random action and observe output
epsilon_random_frames = 3000
# Number of frames for exploration
epsilon_greedy_frames = 2000
max_memory_length = 10000000
update_after_actions = 10
update_target_network = 100
# Define the Huber loss function
loss_function = Huber()
best_reward = 0.0
max_step_per_episode = 30  # Example value, adjust as needed



# Initialize variables
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
# get 10 GBPUSD D1 bars from the current day
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 40000)
df = pd.DataFrame(rates)
print(df)

indicator_bb_10 = BollingerBands(close=df["close"], window=10, window_dev=3)
# Add Bollinger Bands features
df['bb_bbm_10'] = indicator_bb_10.bollinger_mavg()
df['bb_bbh_10'] = indicator_bb_10.bollinger_hband()
df['bb_bbl_10'] = indicator_bb_10.bollinger_lband()

# Add Bollinger Band high indicator
df['bb_bbhi_10'] = indicator_bb_10.bollinger_hband_indicator()

# Add Bollinger Band low indicator
df['bb_bbli_10'] = indicator_bb_10.bollinger_lband_indicator()

# Initialize Bollinger Bands Indicator
indicator_bb_20 = BollingerBands(close=df["close"], window=30, window_dev=2)
# Add Bollinger Bands features
df['bb_bbm_20'] = indicator_bb_20.bollinger_mavg()
df['bb_bbh_20'] = indicator_bb_20.bollinger_hband()
df['bb_bbl_20'] = indicator_bb_20.bollinger_lband()

# Add Bollinger Band high indicator
df['bb_bbhi_20'] = indicator_bb_20.bollinger_hband_indicator()

# Add Bollinger Band low indicator
df['bb_bbli_20'] = indicator_bb_20.bollinger_lband_indicator()

indicator_bb_50 = BollingerBands(close=df["close"], window=50, window_dev=3)
# Add Bollinger Bands features
df['bb_bbm_50'] = indicator_bb_50.bollinger_mavg()
df['bb_bbh_50'] = indicator_bb_50.bollinger_hband()
df['bb_bbl_50'] = indicator_bb_50.bollinger_lband()

# Add Bollinger Band high indicator
df['bb_bbhi_50'] = indicator_bb_50.bollinger_hband_indicator()

# Add Bollinger Band low indicator
df['bb_bbli_50'] = indicator_bb_50.bollinger_lband_indicator()

# Drop specific columns
columns_to_drop = ['tick_volume','spread', 'real_volume', 'time']  # Replace with actual column names
df = df.drop(columns=columns_to_drop)

# Clean NaN values
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)
# df['time']=pd.to_datetime(df['time'], unit='s')
# df['time']=df['time'].dt.hour/24
df['buy_profit'] = 0.0
df['sell_profit'] = 0.0
# print(df.iloc[10]['close'])
# print(df.iloc[10])

class RealForex:
    
    def __init__(self, max_step_per_episode=20, lot=0.01):
        global episode_reward
        super(RealForex, self).__init__()
        self.max_step_per_episode = max_step_per_episode
        self.lot = lot*point
        self.tp = 100
        self.action_space = spaces.Discrete(6)
        self.price_action = []
        self.steps = 0
        self.start_point = 0
        self.market = 0.0
        self.states = np.zeros(21, dtype=np.float32)
        self.len = len(df)- max_step_per_episode

    def reset(self):
        # global episode_reward
        self.steps = 0
        self.price_action = []
        self.start_point = random.randint(0, self.len-1)
        # self.market = df.iloc[self.start_point]['close']
        # state = self.get_state(self.start_point)
        # print(f'states: {state}')
        # return state

    def step(self, action):
        global episode_reward
        info = "Ramin Madani developed this Environment"
        done = False
        reward = 0
        
        # Get current state
        self.states = self.get_state(self.start_point)
        self.market = self.states['close']
        # print(f'states:\n {self.states}')

        # market movement
        self.start_point +=1

        

        # Execute action
        reward , next_state = self.execute_action(action,self.market )

        if len(self.price_action) == 0:
            done = True
            # episode_reward = reward
            
        return next_state, reward, done, info
    
    def get_state(self, nth_point):
        states = df.iloc[nth_point]
        buy_profit, sell_profit = self.calculate_profits(states['close'])
        states['buy_profit'] = buy_profit
        states['sell_profit'] = sell_profit
        return states

    def calculate_profits(self, close):
        buy_profit = sum(self.lot *(close - item[1]) for item in self.price_action if item[0] == 1)
        sell_profit = sum(self.lot * (item[1] - close) for item in self.price_action if item[0] == 2)
        return buy_profit, sell_profit

    def execute_action(self, action, before_price):
        reward = 0
        
        if action == 0:  #0 = nothing
            # reward = next_state['buy_profit']+next_state['sell_profit'] - self.states['buy_profit'] - self.states['sell_profit']
            reward = sum(self.lot * (self.market - item[1]) if item[0] == 1 else self.lot * (item[1] - self.market) for item in self.price_action)
            self.steps += 1
        elif action == 1:  #1 = buy
            self.price_action.append([1, before_price])
            # reward = self.lot * (next_state['close'] - before_price)
            
            reward += 0.02 # living penalty
            self.steps += 1
        elif action == 2:  #2 = sell
            self.price_action.append([2, before_price])
            # reward = self.lot  * (before_price - next_state['close'] )
            
            reward += 0.02 # living penalty
            self.steps += 1
        elif action == 3:  #3 = close all
            reward = self.close_all_positions()
            # self.reset()
        elif action == 4:  #4 = close buy
            reward = self.close_positions(1)
        elif action == 5:   #5 = close sell
            reward = self.close_positions(2)
        # print(f'Action: {action}  , Reward: {reward}')
        if reward<0:
            reward *= 5
        # Get next state
        next_state = self.get_state(self.start_point)
        self.market = next_state['close']
        if action in [0, 1]:
            reward += sum(self.lot * (self.market - item[1]) if item[0] == 1 else self.lot * (item[1] - self.market) for item in self.price_action)
        # print(f'reward: {reward:0.2f}')
        # print(f'next_state:\n {next_state}')
        return reward, next_state

    def close_all_positions(self):
        reward = sum(self.lot * (self.market - item[1]) if item[0] == 1 else self.lot * (item[1] - self.market) for item in self.price_action)
        self.price_action.clear()
        return reward

    def close_positions(self, position_type):
        reward = 0
        remaining_positions = []
        for item in self.price_action:
            if item[0] == position_type:
                reward += self.lot * (self.market - item[1]) if position_type == 1 else self.lot * (item[1] - self.market)
            else:
                remaining_positions.append(item)
        self.price_action = remaining_positions
        self.steps += 1
        
        return reward
    
    ''' 0 = nothing
    1 = buy
    2 = sell
    3 = close all
    4 = close buy
    5 = close sell'''

# Initialize environment
env = RealForex(max_step_per_episode=max_step_per_episode)  # Example environment, replace with actual
num_actions = env.action_space.n
print(num_actions)

# Define the model
def create_q_model():
    # Network defined by the Deepmind paper
    model = keras.Sequential()
    model.add(layers.Input(shape=(len(env.states),)))
    # model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    # model.add(layers.Dense(512, activation="relu"))
    # model.add(layers.Dense(500, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_actions, activation="softmax"))  # Use linear activation for Q-values
    return model
# The first model makes the predictions for Q-values which are used to
# make a action.


# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
if os.path.exists("./saved_models/best_model.h5"):
    model        = keras.models.load_model('./saved_models/best_model.h5')
    target_model = keras.models.load_model('./saved_models/best_model.h5')
else:
    model = create_q_model()
    target_model = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Compile the model manually
model.compile(optimizer=optimizer, loss=loss_function)
target_model.compile(optimizer=optimizer, loss=loss_function)

# episode_reward = 0

def run_episode():
    global frame_count, running_reward, episode_count, best_reward, epsilon
    env.reset()
    epi_reward = 0

    for timestep in range(0, max_step_per_episode):
        frame_count += 1

        
        if env.steps==0:
            action = random.randint(0,1)
        elif env.steps >= env.max_step_per_episode-1:
            action = 2
        elif frame_count < epsilon_random_frames or epsilon > np.random.rand():
            # print('epsilon_random_frames')
            action = np.random.choice(num_actions)
        else:
            state= env.states
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action = np.argmax(model.predict(state_tensor))

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        next_state, reward, done, info = env.step(action)
        # print(f'Next State: {next_state}')

        print(f'Step: {env.steps-1}  , Action: {action}  , Reward: {reward:.2f}  , Done: {done}')

        # Store experience
        state_next_history.append(next_state)
        rewards_history.append(reward)
        done_history.append(done)

        # Update state
        env.states = next_state
        if action in [4, 5]:
            epi_reward += reward
        if done:
            epi_reward+=reward
            env.reset()
            break
            

    # running_reward =  np.mean(rewards_history[-300:])
    if episode_count > 1000:
        running_reward += epi_reward if epi_reward > 0 else epi_reward/5
    episode_count += 1

    # # Save model if best reward is achieved
    if epi_reward > best_reward:
        best_reward = epi_reward
        # Save best_reward to file:
        model.save('./saved_models/best_model.h5')
        save_number(best_reward, 'best_reward.txt')

    return epi_reward

# check 2 models are equal
# def are_models_equal(model1, model2):
#     weights1 = model1.get_weights()
#     weights2 = model2.get_weights()

#     if len(weights1) != len(weights2):
#         return False

#     for w1, w2 in zip(weights1, weights2):
#         if not np.array_equal(w1, w2):
#             return False
    
#     return True

while True:
    episode_reward = run_episode()
    print(f"Episode: {episode_count}, episode_reward: {episode_reward:.4f}, Running Reward: {running_reward:.4f}")

    # Update target network
    if frame_count % update_target_network == 0:
        target_model.set_weights(model.get_weights())
        # print(f'Are models equal: {are_models_equal(model, target_model)}')

    # Train model

    if frame_count % update_after_actions == 0 and len(state_next_history) > batch_size:
        indices = np.random.choice(range(len(state_next_history)), size=batch_size)
        state_sample = np.array([state_next_history[i] for i in indices])
        rewards_sample = [rewards_history[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

        future_rewards = target_model.predict(state_sample)
        updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

        masks = tf.one_hot(indices, num_actions)
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)
            print(f'LOSS: {loss}')
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))