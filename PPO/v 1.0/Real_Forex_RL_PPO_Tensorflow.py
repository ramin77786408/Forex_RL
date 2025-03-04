# from locale import normalize
import os       
import time
import numpy as np
import pandas as pd
from environment import TrainForex, RealForex, get_data_bollinger, get_data_ichimoku, get_data_MACD_STOCH
from Agent import PPOAgent, PPOAgent_Tuner, ReplayBuffer
import MetaTrader5 as mt5
import tensorflow as tf
import multiprocessing

# Set environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(f"Using GPU: {gpus}")
# if gpus:
#     try:
#         # Set memory growth to avoid TensorFlow from consuming all GPU memory
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"Using GPU: {gpus}")
#     except RuntimeError as e:
#         print(e)

buffer_capacity = 1000000
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1  # 15-minute timeframe
data_mode = "Bollinger" # Ichimuko or Bollinger or MACD_stoch
path = f'PPO/v 1.0/Saved Models/model_{symbol}_{timeframe}_{data_mode}_'
num_episodes = 20000
max_step_per_episode = 20  # Example value, adjust as needed
episode_reward = 0.0
running_reward = 0.0
best_reward = 0.0
actions_row = []
ichimuko_shift = 27
# Set random seeds for reproducibility
# np.random.seed(42)

# Set option to display all columns
pd.set_option('display.max_columns', None)
# This approach ensures that all columns of the DataFrame are visible when printed in the terminal.
pd.reset_option('display.max_columns')

# Get data
if data_mode=="Ichimuko":
    df = get_data_ichimoku(symbol, timeframe,number_of_candles=100000, shift=ichimuko_shift) #,max_period=25)
elif data_mode=="Bollinger":
    df = get_data_bollinger(symbol,timeframe,number_of_candles=100000)
elif data_mode=="MACD_stoch":
    df = get_data_MACD_STOCH(symbol,timeframe,number_of_candles=100000)

# Initialize train environment
env = TrainForex(max_step_per_episode=max_step_per_episode, df = df)  # Example environment, replace with actual
num_actions = env.action_space.n
print(num_actions)


state_dim = env.states.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(state_dim, action_dim)
if os.path.exists('PPO/v 1.0/Saved Models/'):
    agent.load_model(path)
replay_buffer = ReplayBuffer(buffer_capacity)

actor_log_dir = 'logs/gradient_tape/' +  '/actor'
critic_log_dir = 'logs/gradient_tape/' +  '/critic'
actor_summary_writer = tf.summary.create_file_writer(actor_log_dir)
critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)

# Training loop

def train_agent(agent, env, replay_buffer, num_episodes, max_step_per_episode, running_reward, actions_row):
    global best_reward
    for episode in range(1,num_episodes):
        state = env.reset()
        episode_reward = 0
        env.reward = 0.0
        env.done = False
        done = False
        while True:
            if env.steps == 0:
                action = np.random.randint(1, 3)
            # if env.steps < max_step_per_episode:
            #     action = 2

            elif len(env.price_action) >= max_step_per_episode:
                action = 3
            else:
                action = agent.get_action(state)

            actions_row.append(action)
            next_state, reward, done, _ = env.step(action)
            if reward>0:
                reward /=1
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                episode_reward = env.reward
                running_reward += episode_reward
                print(f"{BLUE}Episode: {episode}, Reward: {episode_reward:.2f}  Running Reward: {running_reward:.2f}  Actions: {actions_row} {len(actions_row)} {RESET}") 
                agent.train(replay_buffer)
                with actor_summary_writer.as_default():
                    tf.summary.scalar('loss', agent.actor_loss.result(), step=episode)
                    tf.summary.scalar('accuracy', agent.actor_accuracy.result(), step=episode)
                with critic_summary_writer.as_default():
                    tf.summary.scalar('loss', agent.critic_loss.result(), step=episode)
                    tf.summary.scalar('accuracy', agent.critic_accuracy.result(), step=episode)

                actions_row.clear()  
                break
        # Reset metrics every epoch
        agent.actor_loss.reset_state()
        agent.critic_loss.reset_state()
        agent.actor_accuracy.reset_state()
        agent.critic_accuracy.reset_state()
        if running_reward > (best_reward + 20):
            agent.save_model(path) 
            best_reward = running_reward
            time.sleep(100)     
        if episode % 200 ==0:
            time.sleep(100)
            
##############################################################################################
# Tuner

# hp = keras_tuner.HyperParameters()
# agent_tuner = PPOAgent_Tuner(state_dim, action_dim, replay_buffer,hp)

# actor = agent_tuner.actor
# print(actor.summary())
# tuner = keras_tuner.RandomSearch(
#     hypermodel=None,
#     objective=None,
#     max_trials=3,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="my_dir",
#     project_name="helloworld",
# )
# print(tuner.search_space_summary())
#################################################################################


# num_threads = 4  # Number of threads to run
# threads = []
# for _ in range(num_threads):
#     thread = threading.Thread(target=train_agent, args=(agent, env, replay_buffer, num_episodes, max_step_per_episode, running_reward, actions_row))
#     threads.append(thread)
#     thread.start()
# # Wait for all threads to complete
# for thread in threads:
#     thread.join()5446


###################################################################################
# play real forex


def play_real_forex(agent, replay_buffer, symbol, timeframe, ichimuko_shift):
    running_reward = 0.0
    real_env = RealForex(symbol=symbol, time_frame=timeframe, lot=0.1, state_dim=len(df.loc[0]))
    state = real_env.reset()
    done = False    
    while True:
        if real_env.check_new_bar():
            if data_mode=="Ichimuko":
                state = get_data_ichimoku(symbol, timeframe,number_of_candles=100, shift=ichimuko_shift).iloc[-2]
            elif data_mode=="Bollinger":
                state = get_data_bollinger(symbol,timeframe,number_of_candles=300).iloc[-2]
            elif data_mode=="MACD_stoch":
                state = get_data_MACD_STOCH(symbol,timeframe,number_of_candles=300).iloc[-2]
                
            # print(state)
            action = agent.get_action(state)
            next_state, reward, done, _ = real_env.step(action)
            # replay_buffer.push(state, action, reward, next_state, done)
            # state = next_state
            if done:
                running_reward += reward
                print(f"Reward: {reward:.2f}  Running Reward: {running_reward:.2f}  Actions: {actions_row}") 
                agent.load_model(path)

##########################################################################################
    
if __name__ == "__main__":
    # Create processes
    p1 = multiprocessing.Process(target=train_agent, args=(agent, env, replay_buffer, num_episodes, max_step_per_episode, running_reward, actions_row))
    p2 = multiprocessing.Process(target=play_real_forex, args=(agent, replay_buffer, symbol, timeframe, ichimuko_shift))

    # Start processes
    p1.start() 
    p2.start()

    # Wait for processes to complete
    p1.join()
    p2.join()
