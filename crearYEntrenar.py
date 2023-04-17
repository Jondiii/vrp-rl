from stable_baselines3 import PPO, DQN
from vrpEnv import VRPEnv
import os
import time

ALGORTIHM = "PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"

ITERATIONS = 20
TIMESTEPS = 2048*20 # Poner múltiplos de 2048

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

## python crearYEntrenar.py >> log_file 2>> err_file --> para ver si falla y para o quçe
env = VRPEnv(multiTrip = True)
env.createEnv(nVehiculos = 30, nNodos = 100, maxNodeCapacity = 4, sameMaxNodeVehicles=True)
env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

start_time = time.time()

for i in range(1, ITERATIONS+1):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

env.render()

env.close()


"""
Nota: usar el comando python crearYEntrenar.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""