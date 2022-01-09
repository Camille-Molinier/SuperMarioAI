import time
start = time.time()

import os
os.system('cls')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# import warnings
# warnings.filterwarnings('ignore')

print("Modules importation :\n")
print(f"{'    Environement modules' :-<50}", end="")
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
print('Done')

print(f"{'    Preprocessing modules' :-<50}", end="")
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt
print('Done\n')

print(f"{'    Reinforcement learning modules' :-<50}", end="")
from stable_baselines3 import PPO
from TrainAndLoggingCallback import TrainAndLoggingCallback
print('Done\n')

################################################################################
#                                 Setup Mario                                  #
################################################################################
print('Setup Mario : ')
env = gym_super_mario_bros.make('SuperMarioBros-v0')
print(f'     - Non wrapped environement actions : {env.action_space}')

env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(f'     - Wrapped environement actions : {env.action_space}\n')

print(f'     - Evironement frames size : {env.observation_space.shape}\n')

print(f'     - Evironement possible movements : {SIMPLE_MOVEMENT}')

##### Game loop #####
# Create a flag - restart or not
done = True

def loop():
    # loop for every single frame of the game
    for step in range(100000):
        if done:
            # Start the game
            env.reset()
        
        # Do random actions
        state, reward, done, info = env.step(env.action_space.sample())
        # Show the game
        env.render()

        # Close the game
        env.close()

# loop()
## NB : env.action_space.sample() => Random action number in SIMPLE_MOVEMENT


################################################################################
#                            Preprocess Environement                           #
################################################################################
print('\nPreprocess Environement : ')
state = env.reset()

plt.figure()
plt.imshow(state)
plt.savefig('./fig/Normal_environement.png')

# Grayscale
env = GrayScaleObservation(env, keep_dim=True)
state = env.reset()

plt.figure()
plt.imshow(state)
plt.savefig('./fig/Grayscaled_environement.png')

# Warp inside the Dummy environement
env = DummyVecEnv([lambda:env])
state = env.reset()
print('     - Dummy environement shape :', state.shape)

# Stack the frames
env = VecFrameStack(env, 4, channels_order='last')
state = env.reset()
print('     - Vector frame stack environement shape :', state.shape)

state = env.reset()

plt.figure()
plt.imshow(state[0])
plt.savefig('./fig/Vector_frame_stack_environement.png')


################################################################################
#                              Train the RL model                              #
################################################################################
print('\nModel training :')
CHECKPOINT_DIR = './train'
LOG_DIR = './logs'

callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

# Building model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

# Train model
model.learn(total_timesteps=1000000, callback=callback)

################################################################################
#                                 Test it out                                  #
################################################################################
# print('\nTest :')

model = PPO.load('./best_model_100000')

while True :

  action, _state = model.predict(state)
  state, reward, done, info = env.step(action)
  env.render()

print(f'\nProcessing complete (time : {round(time.time()-start, 4)}s)')
