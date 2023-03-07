from stable_baselines3 import PPO
from vrpEnv import VRPEnv
import os
import time

ALGORTIHM = "PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = VRPEnv(nVehiculos = 5, nNodos = 20, twMax = 20)

env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

ITERATIONS = 10
TIMESTEPS = 100

start_time = time.time()

for i in range(1, ITERATIONS):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

env.render()

env.close()