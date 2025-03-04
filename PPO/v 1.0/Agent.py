import keras_tuner
import numpy as np
import random
import time
import tensorflow as tf
import keras
from keras import layers
import os
os.environ["KERAS_BACKEND"] = "tensorflow"


# random.seed(42)
# Hyperparameters
gamma = 0.99
clip_ratio = 0.2
actor_lr = 1.0
critic_lr = 1.0
batch_size = 512
num_epochs = 20

GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 1e-9
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = keras.optimizers.Adadelta(actor_lr)
        self.critic_optimizer = keras.optimizers.Adadelta(critic_lr)
        self.actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        self.actor_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('actor_accuracy')
        self.critic_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
        self.critic_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('critic_accuracy')
        
    def build_actor(self):
        keras.regularizers.L2(0.001)
        state_input = layers.Input(shape=(self.state_dim,))
        dense1 = layers.Dense(256,kernel_regularizer='L2', activation='tanh')(state_input)
        dense2 = layers.Dense(256,kernel_regularizer='L2', activation='softplus')(dense1)
        dense3 = layers.Dense(128,kernel_regularizer='L2', activation='tanh')(dense2)
        # dense4 = layers.Dense(256,kernel_regularizer='L1L2', activation='selu')(dense3)
        output = layers.Dense(self.action_dim, activation='softmax')(dense3)
        model = keras.Model(inputs=state_input, outputs=output)
        # dense1.trainable = False
        return model

    def build_critic(self):
        keras.regularizers.L2(0.001)
        state_input = layers.Input(shape=(self.state_dim,))
        dense1 = layers.Dense(256,kernel_regularizer='L2', activation='tanh')(state_input)
        dense2 = layers.Dense(256,kernel_regularizer='L2', activation='softplus')(dense1)
        dense3 = layers.Dense(128,kernel_regularizer='L2', activation='linear')(dense2)
        # dense4 = layers.Dense(256,kernel_regularizer='L1L2', activation='softplus')(dense3)
        output = layers.Dense(1)(dense3)
        model = keras.Model(inputs=state_input, outputs=output)
        # dense1.trainable = False
        return model
    
    def save_model(self, filepath, **kwargs):
        self.actor.save(filepath +"actor.keras")
        self.critic.save(filepath +"critic.keras")
        print(f"{GREEN}Model saved to {filepath}{RESET}")

    def load_model(self, filepath, **kwargs):
        try:
            self.actor = keras.models.load_model(filepath +"actor.keras")
            self.critic = keras.models.load_model(filepath +"critic.keras")
            print(f"{GREEN}Model loaded from {filepath}{RESET}")
        except Exception as e:
            print(f"An error occurred during LOAD Model: {e}")
        
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=action_probs.numpy()[0])
        return action

    def normalizer(self, data):
        normalizer_layer = tf.keras.layers.Normalization(axis=-1)
        normalizer_layer.adapt(data)
        return normalizer_layer(data).numpy()
    

    def train(self, replay_buffer):
        if len(replay_buffer) < batch_size:
            return
        
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        for epoch in range(num_epochs):
            with tf.GradientTape(persistent=True) as tape:
                # Critic loss
                target_q = reward + gamma * (1 - done) * self.critic(next_state)
                current_q = self.critic(state)
                critic_losses = tf.reduce_mean(tf.square(current_q - target_q))

                # Actor loss
                action_probs = self.actor(state)
                action_one_hot = tf.one_hot(action, self.action_dim)
                selected_action_probs = tf.reduce_sum(action_probs * action_one_hot, axis=1)
                old_action_probs = tf.stop_gradient(selected_action_probs) + self.epsilon
                ratio = selected_action_probs / old_action_probs
                surr1 = ratio * reward
                surr2 = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * reward
                actor_losses = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Update networks
            actor_grads = tape.gradient(actor_losses, self.actor.trainable_variables)
            critic_grads = tape.gradient(critic_losses, self.critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
            
            self.actor_loss(actor_losses)
            self.critic_loss(critic_losses)
            self.actor_accuracy(surr1,surr2)
            self.critic_accuracy(target_q,current_q)
        print(f'{YELLOW}actor_loss:{actor_losses:.5f}  critic_loss:{critic_losses:.5f}{RESET}')

###############################################################################################


# PPO Agent
class PPOAgent_Tuner(keras_tuner.HyperModel):
    def __init__(self, state_dim, action_dim, replay_buffer, hp ):
        self.hp = hp
        self.replay_buffer=replay_buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon = 1e-9
        self.actor = self.build("actor",self.hp)
        self.critic = self.build("critic",self.hp)
        self.actor_optimizer = keras.optimizers.Adam(self.hp.Float(name="actor_lr", min_value=1e-6, max_value=1e-1, sampling="log"))
        self.critic_optimizer = keras.optimizers.Adam(self.hp.Float(name="critic_lr", min_value=1e-6, max_value=1e-1, sampling="log"))


    def build(self, name,hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.state_dim,)))
        # Tune the number of layers.
        number_of_layer = hp.Int(name="num_layer",min_value=1,max_value=6)
        for i in range(number_of_layer):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(name=f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice(name="activation", values=["relu", "tanh"]),
                )
            )
        if hp.Boolean(name="dropout"):
            model.add(layers.Dropout(rate=0.25))
        if name=="actor":
            model.add(layers.Dense(self.action_dim, activation="softmax"))
        if name=="critic":
            model.add(layers.Dense(1, activation="softmax"))

        return model
 
    def save_model(self, filepath, **kwargs):
        # Save the model with the best hyperparameters
        actor = self.build_model("actor",keras_tuner.HyperParameters().from_config(kwargs))
        actor.save('actor'+filepath)
        actor.summary()
        critic = self.build_model("critic",keras_tuner.HyperParameters().from_config(kwargs))
        critic.save('critic'+filepath)
        critic.summary()
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.actor = keras.models.load_model('actor'+filepath)
        self.critic = keras.models.load_model('critic'+filepath)
        print(f"Model loaded from {filepath}")
        
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=action_probs.numpy()[0])
        return action

    def train(self, replay_buffer):
        print(f'replay_buffer_ size: {len(replay_buffer)}')
        if len(replay_buffer) < batch_size:
            return
        number_of_train = int(len(replay_buffer)* 1)
        
        state, action, reward, next_state, done = replay_buffer.sample(number_of_train)
        for epoch in range(num_epochs):
            with tf.GradientTape(persistent=True) as tape:
                # Critic loss
                target_q = reward + gamma * (1 - done) * self.critic(next_state)
                current_q = self.critic(state)
                critic_loss = tf.reduce_mean(tf.square(current_q - target_q))

                # Actor loss
                action_probs = self.actor(state)
                action_one_hot = tf.one_hot(action, self.action_dim)
                selected_action_probs = tf.reduce_sum(action_probs * action_one_hot, axis=1)
                old_action_probs = tf.stop_gradient(selected_action_probs) + self.epsilon
                ratio = selected_action_probs / old_action_probs
                surr1 = ratio * reward
                surr2 = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * reward
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                # print(f'{YELLOW}actor_loss:{surr2} {surr1} critic_loss:{target_q}{RESET}')

            # Update networks
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
        
        print(f'actor_loss:{actor_loss:.5f}  critic_loss:{critic_loss:.5f}')
        return {"critic_loss":critic_loss, "actor_loss":actor_loss}

# **************************************************************************************** #


# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)