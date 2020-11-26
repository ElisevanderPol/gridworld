from gym.envs.registration import register

register(
    id='GridEnv-v1',
    entry_point='gridworld.envs:GridEnv',
)
