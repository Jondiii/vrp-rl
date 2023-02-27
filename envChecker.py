from vrpEnv import VRPEnv

from stable_baselines3.common.env_checker import check_env

env = VRPEnv(nVehiculos = 2, nNodos=5)
# It will check your custom environment and output additional warnings if needed
check_env(env)