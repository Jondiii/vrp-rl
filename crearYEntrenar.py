from stable_baselines3 import PPO, DQN
from vrpEnv import VRPEnv
import os
import time

ALGORTIHM = "PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"


ITERATIONS = 5
TIMESTEPS = 2048*2 # Poner múltiplos de 2048


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

## python3 arduinoDriver.py >> log_file 2>> err_file --> para ver si falla y para o quçe
env = VRPEnv()
env.createEnv(5, 10, sameMaxNodeVehicles=True)
env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)
env.reset()


model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

#checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=models_dir)
print(model.num_timesteps)

start_time = time.time()

for i in range(1, ITERATIONS+1):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print(model.num_timesteps)

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

env.render()

env.close()