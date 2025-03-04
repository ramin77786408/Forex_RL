
"""
این برنامه با محیط کاملا تصادفی که برای فارکس نوشته شده است با استفاده از الگوریتم مینیمکس 
انتخاب به گرفتن پوزیشن میکند . الگوریتم مینیمکس کاملا سالم است و تست شده.
 ولی شاید محیط اجرایی نیاز به اصلاحاتی داشته باشد البته بعید میدونم چون انتخابهای الگوریتم
   با منطق من درست است. محیط بصورت رندوم ۱۰۰پوینت بالا یا پایین میرود و الگوریتم یک بای 
    و یک سل بصورت نوبتی میگیرد برای آشنایی بیشتر با محیط مجازی پایتون در فولدر بالایی میتوان 
    برنامه را اجرا کرد.

"""


import random
import time

# Define color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
CYAN = "\033[36m"
RESET = "\033[0m"

point = 100000
episode_reward = 0.0
running_reward = 0.0
epochs= 20
class TrainForex:
    
    def __init__(self, max_step_per_episode=10, lot=0.1):
        global episode_reward
        super(TrainForex, self).__init__()
        self.max_step_per_episode = max_step_per_episode
        self.lot = lot
        self.tp = 200
        self.action_space = [1, 2, 3, 4]  # Buy, Sell, Buy 2x, Sell 2x
        self.market_movment = ["up", "down"]
        self.steps=0
        self.depth_of_think = 6
        self.price_action = []
        self.market = 10000
        self.market_history = []
        self.reward_of_future=[]

    def reset(self):
        # global episode_reward
        self.steps = 0
        self.price_action = []
        self.market_history = []
        self.market_history.append(self.market)
        
    
    def calculate_profits(self, close):
        buy_profit = sum(item[1] *(close - item[2]) for item in self.price_action if item[0] == 1)
        sell_profit = sum(item[1] * (item[2] - close) for item in self.price_action if item[0] == 2)
        return buy_profit, sell_profit

    def execute_action(self, action, price):
        
        if action == 1:  #1 = buy
            self.price_action.append([1, self.lot,price])
            
        elif action == 2:  #2 = sell
            self.price_action.append([2, self.lot, price])
            
        elif action == 3:  #3 = buy 2x
            self.price_action.append([1, self.lot *2,price])
            
        elif action == 4:  #4 = sell 2x
            self.price_action.append([2, self.lot *2, price])
            
        elif action == "up":  # market goes up
            self.market = self.market_history[-1]
            self.market += self.tp
            self.market_history.append(self.market)
            
        elif action == "down":  #market goes down
            self.market = self.market_history[-1]
            self.market -= self.tp
            self.market_history.append(self.market)
        
    
    # Check if the game is over
    def is_game_over(self, depth):
        return depth > self.depth_of_think



    # Implement the Minimax algorithm:
    def minimax(self, depth, is_maximizing, price):
        if self.is_game_over(depth):
            buy_profit, sell_profit = self.calculate_profits(price)
            self.reward_of_future.append(buy_profit + sell_profit)
            return buy_profit + sell_profit
        

        if is_maximizing:
            best_score = -float('inf')
            for action in self.action_space:
                self.execute_action(action, self.market_history[-1])
                score = self.minimax(depth + 1, False, self.market_history[-1])
                self.price_action.pop()  # Undo the action
                best_score = max(best_score, score)
                # print(f"Maximizing: Depth {depth}, Action {action}, Score {score}, Best Score {best_score}")
                # time.sleep(5)
            return best_score
        else:
            best_score = float('inf')
            for action in self.market_movment:
                self.execute_action(action, self.market_history[-1])
                score = self.minimax(depth + 1, True, self.market_history[-1])
                self.market_history.pop()  # Undo the action
                best_score = min(best_score, score)
                # print(f"Minimizing: Depth {depth}, Action {action:>4}, Score {score}, Best Score {best_score}")
                # time.sleep(5)
            return best_score

    def find_best_move(self, price):
        best_score = -float('inf')
        best_action = None
        for action in self.action_space:
            self.execute_action(action, price)
            score = self.minimax(0, False, price)
            self.price_action.pop()  # Undo the action
            if score > best_score:
                best_score = score
                best_action = action
        # print(f'self.reward_of_future \n{self.reward_of_future}')
        # print(len(self.reward_of_future))
        return best_action, best_score
    
    def step(self, action):
        self.execute_action(action, self.market_history[-1])
        self.steps+=1
        market_movement = random.choice(env.market_movment)
        env.execute_action(market_movement, env.market_history[-1])
        buy_profit, sell_profit = self.calculate_profits(self.market_history[-1])
        reward = buy_profit + sell_profit
        done = self.steps > self.max_step_per_episode
        return market_movement, reward, done
    
# Example usage
env = TrainForex(max_step_per_episode=20)
for episode in range(1,epochs):
    env.reset()
    price = env.market  # Example price, replace with actual
    best_action , best_score= env.find_best_move(price)
    print(f"Episode: {episode:>2}  Best action: {best_action}  Best score: {best_score}")
    while env.steps<=20:
        market_movement, reward, done = env.step(best_action)
        if done:
            running_reward += reward
            print(f"{YELLOW}Epoch:{episode:<3}---->>>  last_Reward: {GREEN if reward >0 else RED}{reward}    {YELLOW} running_reward: {RED if running_reward <= 0 else GREEN}{running_reward}{RESET}")
            break
        best_action, best_score = env.find_best_move(env.market_history[-1])
        print(f"Step: {env.steps:<3}{GREEN}-->>{RESET}  Action: {best_action}   Market movement: {market_movement:<5}   Reward: {reward:>4}")
    
        time.sleep(1)

