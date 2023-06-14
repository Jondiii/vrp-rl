from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN, A2C
#from wakepy import keepawake
from vrpEnv import VRPEnv
import os
import time

ALGORTIHM = "A2C_LargoNoMultiTrip"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"

ITERATIONS = 50
TIMESTEPS = 2048*10 # Poner múltiplos de 2048

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


env = VRPEnv()
env.createEnv(nVehiculos = 5, nNodos = 20, maxNodeCapacity = 4, sameMaxNodeVehicles=True)
env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)
env.reset()

model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

start_time = time.time()
#with keepawake(keep_screen_awake=False):

for i in range(1, ITERATIONS+1):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

env.render()

env.close()


"""
Nota: usar el comando python crearYEntrenar.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""
