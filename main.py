from stable_baselines3 import PPO
from vrpEnv import VRPEnv
import os

ALGORTIHM = "PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = VRPEnv(nVehiculos = 2, nNodos = 10, multiTrip = True)

env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

ITERATIONS = 10
TIMESTEPS = 1000

for i in range(1, ITERATIONS):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.render()

env.close()