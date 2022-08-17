from gym.envs.registration import register

register(
    id = 'dpath-v0',
    entry_point = 'gym_hy2.envs:Agent')