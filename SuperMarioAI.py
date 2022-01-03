import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

################################################################################
#                                 Setup Mario                                  #
################################################################################
print('\nSetup Mario :')
env = gym_super_mario_bros.make('SuperMarioBros-v0')
print(f'     - Non wrapped environement actions : {env.action_space}')

env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(f'     - Wrapped environement actions : {env.action_space}\n')

print(f'     - Evironement frames size : {env.observation_space.shape}\n')

print(f'     - Evironement possible movements : {SIMPLE_MOVEMENT}\n')

##### Game loop #####
# Create a flag - restart or not
done = True

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

## NB : env.action_space.sample() => Random action number in SIMPLE_MOVEMENT

################################################################################
#                            Preprocess Environement                           #
################################################################################

################################################################################
#                              Train the RL model                              #
################################################################################

################################################################################
#                                 Test it out                                  #
################################################################################