from asyncio import exceptions
from sys import exception
from gym import spaces
import numpy as np
import random
import MetaTrader5 as mt5
from ta.volatility import BollingerBands
import pandas as pd
from datetime import datetime
import time
import pytz

RED = "\033[31m"
RESET = "\033[0m"

point = 100000
episode_reward = 0.0
running_reward = 0.0
timezone = pytz.timezone("Asia/Calcutta")


class TrainForex:
    
    def __init__(self, max_step_per_episode=30, lot=0.1, df=None):
        global episode_reward
        super(TrainForex, self).__init__()
        self.max_step_per_episode = max_step_per_episode
        self.lot = lot*point
        self.tp = 100
        self.action_space = spaces.Discrete(6)
        self.price_action = []
        self.steps = 0
        self.df = df
        self.reward=0.0
        self.done = False
        self.start_point = 0
        self.market = 0.0
        self.states = np.zeros(df.shape[1], dtype=np.float64)
        self.len = len(self.df)- max_step_per_episode-1

    def reset(self):
        # global episode_reward
        self.steps = 0
        self.reward=0.0
        self.price_action = []
        self.start_point = random.randint(0, self.len-2)
        self.market = self.df.iloc[self.start_point]['close']
        state = self.get_state(self.start_point)
        # print(f'states: {state}')
        return state

    def step(self, action):
        global episode_reward, running_reward
        info = "Ramin Madani developed this Environment"
        call_margin = False
        # Get current state
        self.states = self.get_state(self.start_point)
        # print(self.states)
        self.market = self.states['close']
        if self.states['profit']< -50:
            call_margin = True

        # market movement
        self.start_point +=1

        # Execute action
        if call_margin:
            print(f'{RED}Call Margin{RESET}')
            self.done=True
            self.reward = self.states['buy_profit'] + self.states['sell_profit']
            next_state = self.states
        else:
            next_state = self.execute_action(action,self.market )

            if self.done:
                episode_reward += self.reward
                running_reward += episode_reward
            else:
                if action in [4,5]:
                    episode_reward += self.reward
                    # running_reward += self.reward
                elif action in [0]:
                    close = next_state['close']
                    buy_profit, sell_profit = self.calculate_profits(close)
                    self.reward +=  (buy_profit+sell_profit)

        return next_state, self.reward, self.done, info
    
    def get_state(self, nth_point):
        try:
            state = self.df.iloc[nth_point]
        except exception as e:
            print(f"last candle formed. {e}")
            self.done = True
        buy_profit, sell_profit = self.calculate_profits(state['close'])
        state['buy_profit'] = buy_profit
        state['sell_profit'] = sell_profit
        state['profit'] = buy_profit + sell_profit
        return state

    def calculate_profits(self, close):
        buy_profit = sum(self.lot *(close - item[1]) for item in self.price_action if item[0] == 1)
        sell_profit = sum(self.lot * (item[1] - close) for item in self.price_action if item[0] == 2)
        return buy_profit, sell_profit

    def execute_action(self, action, before_price):
        self.market = self.get_state(self.start_point)['close']
        if action == 0:  #0 = nothing
            before_profit = self.calculate_profits(before_price)[0] + self.calculate_profits(before_price)[1]
            now_profit = self.calculate_profits(self.market)[0] + self.calculate_profits(self.market)[1]
            self.reward = now_profit-before_profit
            self.steps += 1

        elif action == 1:  #1 = buy
            self.price_action.append([1, before_price])
            self.reward = -0.1 # living penalty
            self.reward += round(self.calculate_profits(self.market)[0] + self.calculate_profits(self.market)[1] - self.calculate_profits(before_price)[0] - self.calculate_profits(before_price)[1], 2)
            self.steps += 1

        elif action == 2:  #2 = sell
            self.price_action.append([2, before_price])
            self.reward = -0.1 # living penalty
            self.reward += round(self.calculate_profits(self.market)[0] + self.calculate_profits(self.market)[1] - self.calculate_profits(before_price)[0] - self.calculate_profits(before_price)[1], 2)
            self.steps += 1

        elif action == 3:  #3 = close all
            self.reward = self.close_all_positions(before_price)
            
        elif action == 4:  #4 = close buy
            self.reward = self.close_positions(1,before_price)

        elif action == 5:   #5 = close sell
            self.reward = self.close_positions(2,before_price)

        next_state = self.get_state(self.start_point)
        self.market = next_state['close']
        return next_state

    def close_all_positions(self,befor_price):
        reward = sum(self.lot * (befor_price - item[1]) if item[0] == 1 else self.lot * (item[1] - befor_price) for item in self.price_action)
        self.price_action.clear()
        self.reward = 0.0
        self.done = True
        return reward

    def close_positions(self, position_type,befor_price):
        reward = 0.0
        remaining_positions = []
        for item in self.price_action:
            if item[0] == position_type:
                reward += self.lot * (befor_price - item[1]) if position_type == 1 else self.lot * (item[1] - befor_price)
            else:
                remaining_positions.append(item)
        self.price_action.clear()
        self.price_action = remaining_positions
        self.steps += 1
        if len(remaining_positions) ==0:
            self.done = True
            self.reward = 0.0
        return reward
    
    ''' 0 = nothing
    1 = buy
    2 = sell
    3 = close all
    4 = close buy
    5 = close sell'''


class RealForex:

    def __init__(self, symbol="EURUSD", lot=0.1, max_step_per_episode=20, time_frame=mt5.TIMEFRAME_M15, state_dim=22):
        global episode_reward
        super(RealForex, self).__init__()
        self.max_step_per_episode = max_step_per_episode
        self.lot = lot
        self.action_space = spaces.Discrete(6)
        self.steps = 0
        self.symbol = symbol
        self.time_frame = time_frame
        self.reward=0.0
        self.nubmer_of_candles = 100
        self.states = np.zeros(state_dim, dtype=np.float32)
        self.done = False
        self.shift = 26
    
    def reset(self):
        global episode_reward
        self.steps = 0
        self.reward=0.0
        episode_reward = 0.0
        state = get_data_ichimoku(self.symbol, self.time_frame,self.nubmer_of_candles, shift=self.shift)
        buy_profit, sell_profit = self.calculate_profits()
        state['buy_profit'] = buy_profit
        state['sell_profit'] = sell_profit
        state['profit'] = buy_profit + sell_profit
        return state

    def step(self, action):
        global episode_reward, running_reward
        info = "Ramin Madani developed this Environment"
        call_margin = False
        # Get current state
        self.states = get_data_ichimoku(self.symbol, self.time_frame,self.nubmer_of_candles, shift=self.shift).iloc[-2]
        # print(f'states: {self.states}')
        buy_profit, sell_profit = self.calculate_profits()
        self.states['buy_profit'] = buy_profit
        self.states['sell_profit'] = sell_profit
        self.states['profit'] = buy_profit + sell_profit
        if self.states['profit'] < -50:
            call_margin = True

        # Execute action
        if call_margin:
            print('Call Margin')
            self.done=True
            self.reward = self.states['buy_profit'] + self.states['sell_profit']
            self.close_all_positions()
            next_state = self.states
        else:
            next_state = self.execute_action(action)

            if self.done:
                episode_reward += self.reward
                running_reward += episode_reward
            else:
                if action in [4,5]:
                    episode_reward += self.reward
                elif action in [0]:
                    buy_profit, sell_profit = self.calculate_profits()
                    self.reward +=  (buy_profit+sell_profit)

        return next_state, self.reward, self.done, info

    
    def calculate_profits(self):
        buy_profit=0.0
        sell_profit=0.0
        # establish connection to MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        positions=mt5.positions_get(symbol=self.symbol)
        if positions==None:
            print("No positions on {}, error code={}".format(mt5.last_error(), self.symbol))
        elif len(positions)>0:
            for position in positions:
                if position.type == 1:
                    sell_profit += position.profit
                if position.type == 0:
                    buy_profit += position.profit
        return buy_profit, sell_profit

    def check_new_bar(self):
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        rate = mt5.copy_rates_from_pos(self.symbol, self.time_frame, 0, 2)
        # while rate is None :
        #     print("mt5.copy_rates_from_pos() failed, error code =",mt5.last_error())
        #     rate = mt5.copy_rates_from_pos(self.symbol, self.time_frame, 0, 2)
        #     time.sleep(1)
        #     print("while loop")
        
        last_bar_time = datetime.fromtimestamp(rate[-1]['time'])
        current_time = datetime.now(timezone)
        print(last_bar_time)
        print(current_time)
        if current_time.minute == last_bar_time.minute:
            print(f"{RED}A new bar has formed.{RESET}")
            return True
        else:
            print("No new bar yet.")
            time.sleep(20)
            return False

        return False
    
    def take_position(self, position_type, volume):
        result = None
        # establish connection to the MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        
        # prepare the buy request structure
        
        lot = volume
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(self.symbol, "not found, can not call order_check()")
            mt5.shutdown()
            quit()
        
        # if the symbol is unavailable in MarketWatch, add it
        if not symbol_info.visible:
            print(self.symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(self.symbol,True):
                print("symbol_select({}}) failed, exit",self.symbol)
                mt5.shutdown()
                quit()
        
        point = mt5.symbol_info(self.symbol).point
        if position_type == 0:
            price = mt5.symbol_info_tick(self.symbol).ask
            type = mt5.ORDER_TYPE_BUY
        if position_type == 1:
            price = mt5.symbol_info_tick(self.symbol).bid
            type = mt5.ORDER_TYPE_SELL
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": type,
            "price": price,
            "sl": 0.0 ,
            "tp": 0.0,
            "deviation": deviation,
            "magic": 234000,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        # send a trading request
        while result ==None:
            try:
                result = mt5.order_send(request)
                # check the execution result
                print("1. order_send(): by {} {} lots at {} with deviation={} points".format(self.symbol,lot,price,deviation));
            except Exception as e:
                print(f"An error occurred during episode: {e}")
            if result != None:
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    result_dict=result._asdict()
                    for field in result_dict.keys():
                        print("   {}={}".format(field,result_dict[field]))
                        # if this is a trading request structure, display it element by element as well
                        if field=="request":
                            traderequest_dict=result_dict[field]._asdict()
                            for tradereq_filed in traderequest_dict:
                                print(" traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))
                    mt5.shutdown()
                    quit()
        
    def execute_action(self, action):
        if action == 0:  #0 = nothing
            print('Nothing')
            before_profit = self.calculate_profits()[0] + self.calculate_profits()[1]
            time.sleep(62)
            while not self.check_new_bar():
                time.sleep(50)
            now_profit    = self.calculate_profits()[0] + self.calculate_profits()[1]
            self.reward = now_profit-before_profit
            self.steps += 1

        elif action == 1:  #1 = buy
            print('Buy')
            before_profit = self.calculate_profits()[0] + self.calculate_profits()[1]
            self.take_position(mt5.ORDER_TYPE_BUY, self.lot)
            self.reward = 0.02 # living penalty
            time.sleep(60)
            while not self.check_new_bar():
                time.sleep(50)
            now_profit    = self.calculate_profits()[0] + self.calculate_profits()[1]
            self.reward += round((now_profit-before_profit) ,2)
            self.steps += 1

        elif action == 2:  #2 = sell
            print('Sell')
            before_profit = self.calculate_profits()[0] + self.calculate_profits()[1]
            self.take_position(mt5.ORDER_TYPE_SELL, self.lot)
            self.reward = 0.02 # living penalty
            time.sleep(60)
            while not self.check_new_bar():
                time.sleep(50)
            now_profit    = self.calculate_profits()[0] + self.calculate_profits()[1]
            self.reward += round((now_profit-before_profit) ,2)
            self.steps += 1

        elif action == 3:  #3 = close all
            print('Close All')
            self.reward = self.close_all_positions()
            time.sleep(50)
        elif action == 4:  #4 = close buy
            print('Close Buy')
            self.reward = self.close_positions(mt5.ORDER_TYPE_BUY)
            time.sleep(50)
        elif action == 5:   #5 = close sell
            print('Close Sell')
            self.reward = self.close_positions(mt5.ORDER_TYPE_SELL)
            time.sleep(50)
        next_state = get_data_ichimoku(self.symbol, self.time_frame, self.nubmer_of_candles, shift=self.shift).iloc[-2]
        # print(f'states_nn: {next_state}')
        return next_state

    def close_all_positions(self):
        reward = mt5.account_info().equity - mt5.account_info().balance
        self.reward = 0.0
        # Close all positions
        positions=mt5.positions_get(symbol=self.symbol)
        for position in positions:
          mt5.Close(symbol=self.symbol, ticket=position.ticket)
        self.done = True
        return reward

    def close_positions(self, position_type):
        reward = 0.0
        positions=mt5.positions_get(symbol=self.symbol)
        for position in positions:
          if position.type == position_type:
            reward += position.profit
            mt5.Close(symbol=self.symbol, ticket=position.ticket)
        self.steps += 1
        if len(mt5.positions_get(symbol=self.symbol)) ==0:
            self.done = True
            self.reward = 0.0
        else:
            time.sleep(61)
        return reward
    
    ''' 0 = nothing
    1 = buy
    2 = sell
    3 = close all
    4 = close buy
    5 = close sell'''


def calculate_profits(symbol):
        buy_profit=0.0
        sell_profit=0.0
        # establish connection to MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        positions=mt5.positions_get(symbol=symbol)
        if positions==None:
            print("No positions on {}, error code={}".format(mt5.last_error(), symbol))
        elif len(positions)>0:
            for position in positions:
                if position.type == 1:
                    sell_profit += position.profit
                if position.type == 0:
                    buy_profit += position.profit
        return buy_profit, sell_profit

def get_data_bollinger(symbol, time_frame, number_of_candles):
    # establish connection to MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    # get 10 GBPUSD D1 bars from the current day
    rates = mt5.copy_rates_from_pos(symbol, time_frame, 0, number_of_candles)
    df = pd.DataFrame(rates)
    
    a=[3,15,60,240]
    # Initialize Bollinger Bands Indicator
    for j,number in enumerate(a):  #
        
        a[j] = BollingerBands(close=df["close"], window=number, window_dev=2)
        df[f'bb_bbm_{number}'] = a[j].bollinger_mavg()
        df[f'bb_bbh_{number}'] = a[j].bollinger_hband()
        df[f'bb_bbl_{number}'] = a[j].bollinger_lband()
        df[f'bb_bbhi_{number}'] = a[j].bollinger_hband_indicator()    # Add Bollinger Band high indicator
        df[f'bb_bbli_{number}'] = a[j].bollinger_lband_indicator()    # Add Bollinger Band low indicator

    
    # Drop specific columns
    columns_to_drop = ['tick_volume','spread', 'real_volume']  # Replace with actual column names
    df = df.drop(columns=columns_to_drop)
    df = df.dropna() # Clean NaN values
    df = df.reset_index(drop=True)    # Reset index
    df['time']=pd.to_datetime(df['time'], unit='s')
    df['time']=df['time'].dt.hour/24
    buy_profit, sell_profit = calculate_profits(symbol)
    df['buy_profit'] = buy_profit
    df['sell_profit'] = sell_profit
    df['profit'] = buy_profit + sell_profit
    # print(df)
    mt5.shutdown()
    # Save the DataFrame to a CSV file
    # df.to_csv('data.csv', index=False)
    # Read the DataFrame from the CSV file
    # df = pd.read_csv('PPO/data/data.csv')
    return df


from ta.trend import IchimokuIndicator
def get_data_ichimoku(symbol, time_frame, number_of_candles, shift):
    # establish connection to MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    # get 10 GBPUSD D1 bars from the current day
    rates = mt5.copy_rates_from_pos(symbol, time_frame, 0, number_of_candles)
    df = pd.DataFrame(rates)
 
    indicator_ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
    
    # Create a dictionary to hold new columns
    new_columns = {
        'senkou_span_a': indicator_ichimoku.ichimoku_a(),
        'senkou_span_b': indicator_ichimoku.ichimoku_b(),
        'kijun-sen': indicator_ichimoku.ichimoku_base_line(),
        'tenkan-sen': indicator_ichimoku.ichimoku_conversion_line()
    }
    
    for i in range(shift):
        new_columns[f'senkou_span_a_{i}'] = indicator_ichimoku.ichimoku_a().shift(i)
        new_columns[f'senkou_span_b_{i}'] = indicator_ichimoku.ichimoku_b().shift(i)
        new_columns[f'kijun-sen_{i}'] = indicator_ichimoku.ichimoku_base_line().shift(i)
        new_columns[f'tenkan-sen_{i}'] = indicator_ichimoku.ichimoku_conversion_line().shift(i)
    
    # Concatenate new columns to the DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    # Drop specific columns
    columns_to_drop = ['tick_volume','spread', 'real_volume']  # Replace with actual column names
    df = df.drop(columns=columns_to_drop)
    df = df.dropna() # Clean NaN values
    df = df.reset_index(drop=True)    # Reset index
    df['time']=pd.to_datetime(df['time'], unit='s')
    df['time']=df['time'].dt.hour/24
    buy_profit, sell_profit = calculate_profits(symbol)
    df['buy_profit'] = buy_profit
    df['sell_profit'] = sell_profit
    df['profit'] = buy_profit + sell_profit
    # print(df)
    mt5.shutdown()
    
    # Save the DataFrame to a CSV file
    # df.to_csv('data.csv', index=False)
    # Read the DataFrame from the CSV file
    # df = pd.read_csv('PPO/data/data.csv')
    return df



from ta.trend import MACD
from ta.momentum import StochasticOscillator, kama

def get_data_MACD_STOCH(symbol, time_frame, number_of_candles):
    # establish connection to MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    # get 10 GBPUSD D1 bars from the current day
    rates = mt5.copy_rates_from_pos(symbol, time_frame, 0, number_of_candles)
    df = pd.DataFrame(rates)

    # MACD
    indicator_macd = MACD(close=df['close'])
    df['macd_line'] = indicator_macd.macd()
    df['signal_line'] = indicator_macd.macd_signal()
    df['histogram'] = indicator_macd.macd_diff()

    # Stochastic Oscillator
    for i in [14, 28, 42, 56]:
        indicator_stoch = StochasticOscillator(high=df['high'],low=df['low'], close=df['close'],window=i)
        df[f'stoch_signal_{i}'] = indicator_stoch.stoch_signal()
        df[f'stoch_{i}'] = indicator_stoch.stoch()
    # Kaufman’s Adaptive Moving Average (KAMA)
    for k in [10, 20, 30, 40]: 
        indicator_kama = kama(close=df['close'],window=k, pow1=int(k/3), pow2=k+20)
        df[f'kama_{k}'] = indicator_kama

    # Drop specific columns
    columns_to_drop = ['tick_volume','spread']  # Replace with actual column names
    df = df.drop(columns=columns_to_drop)
    df = df.dropna() # Clean NaN values
    df = df.reset_index(drop=True)    # Reset index
    df['time']=pd.to_datetime(df['time'], unit='s')
    df['time']=df['time'].dt.hour/24
    buy_profit, sell_profit = calculate_profits(symbol)
    df['buy_profit'] = buy_profit
    df['sell_profit'] = sell_profit
    df['profit'] = buy_profit + sell_profit
    # print(df)
    mt5.shutdown()
    
    return df