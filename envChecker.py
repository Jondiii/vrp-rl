from vrpEnv import VRPEnv

from stable_baselines3.common.env_checker import check_env

env = VRPEnv(nNodos=5, nGrafos=6)
# It will check your custom environment and output additional warnings if needed
check_env(env)