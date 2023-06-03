from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN, A2C
#from wakepy import keepawake
from vrpEnv import VRPEnv
import os
import time

ITERATIONS = 500
TIMESTEPS = 2048*5 # Serán 5M de steps

listaAlgoritmos = ["decreasing_DQN_P_2", "increasing_DQN_P_2", "decreasing_DQN_M_2", "increasing_DQN_M_2", "increasing_DQN_G_2", "decreasing_DQN_G_2"]

for algoritmo in listaAlgoritmos:
    models_dir = "modelsEXP/" + algoritmo
    log_dir = "logsEXP"

    caso = algoritmo.split("_")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if caso[2] == 'P':
        nVehiculos = 5
        nNodos = 20
    
    if caso[2] == 'M':
        nVehiculos = 13
        nNodos = 50

    if caso[2] == 'G':
        nVehiculos = 25
        nNodos = 100
    
    env = VRPEnv(multiTrip = True)
    env.createEnv(nVehiculos = nVehiculos, nNodos = nNodos, maxNodeCapacity = 4, sameMaxNodeVehicles=True)
    
    if caso[0] == "decreasing":
        env.setDecayingIsDone(ITERATIONS * TIMESTEPS)

    if caso[0] == "increasing":
        env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)
    env.reset()


    if caso[1] == 'PPO':
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    if caso[1] == 'A2C':
        model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    if caso[1] == 'DQN':
        model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    start_time = time.time()
    #with keepawake(keep_screen_awake=False):

    for i in range(1, ITERATIONS+1):
        model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = algoritmo)
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

    env.render()

    env.close()


    """
    Nota: usar el comando python experimentacion.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
    """
