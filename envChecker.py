from vrpEnv import VRPEnv

from stable_baselines3.common.env_checker import check_env

env = VRPEnv()
env.createEnv(50, 100, sameMaxNodeVehicles=True)

# COMENTARIO DE PRUEBA

# It will check your custom environment and output additional warnings if needed
check_env(env)
