from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO, DQN, A2C
from multiprocessing import Pool
from vrpEnv import VRPEnv
import os
import time

ITERATIONS = 500
TIMESTEPS = 2048*10 # Serán 10M de steps

listaMetodo = ["normal", "increasing", "decreasing"]
listaAlgoritmo = ["PPO", "DQN", "A2C"]
listaTamanyo = ["P", "M", "G"]
listaExpNumber = [*range(1, 21)]


def crearDirectorios(models_dir, log_dir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def crearModelo(algoritmo, env, log_dir):
    if algoritmo == 'PPO':
        return PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu')

    if algoritmo == 'A2C':
        return  A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu')

    if algoritmo == 'DQN':
        return DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu')


def crearEnv(nVehiculos, nNodos, metodo):
    env = VRPEnv() #TODO Con esto las rutas que salen al final son de vehiculos yendo y viniendo del depot con un solo pedido
    env.createEnv(nVehiculos = nVehiculos, nNodos = nNodos, maxNodeCapacity = 4, sameMaxNodeVehicles=True)

    if metodo == "decreasing":
        env.setDecayingIsDone(ITERATIONS * TIMESTEPS)

    if metodo == "increasing":
        env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)

    env.reset()

    return env

#listaPruebas = ['P_PPO_normal_1', 'P_PPO_normal_2']
listaPruebas = ['P_PPO_normal_1', 'P_PPO_normal_2', 'P_PPO_normal_3', 'P_PPO_normal_4', 'P_PPO_normal_5', 'P_PPO_normal_6', 'P_PPO_normal_7', 'P_PPO_normal_8', 'P_PPO_normal_9', 'P_PPO_normal_10', 'P_PPO_normal_11', 'P_PPO_normal_12', 'P_PPO_normal_13', 'P_PPO_normal_14', 'P_PPO_normal_15', 'P_PPO_normal_16', 'P_PPO_normal_17', 'P_PPO_normal_18', 'P_PPO_normal_19', 'P_PPO_normal_20', 'P_PPO_increasing_1', 'P_PPO_increasing_2', 'P_PPO_increasing_3', 'P_PPO_increasing_4', 'P_PPO_increasing_5', 'P_PPO_increasing_6', 'P_PPO_increasing_7', 'P_PPO_increasing_8', 'P_PPO_increasing_9', 'P_PPO_increasing_10', 'P_PPO_increasing_11', 'P_PPO_increasing_12', 'P_PPO_increasing_13', 'P_PPO_increasing_14', 'P_PPO_increasing_15', 'P_PPO_increasing_16', 'P_PPO_increasing_17', 'P_PPO_increasing_18', 'P_PPO_increasing_19', 'P_PPO_increasing_20', 'P_PPO_decreasing_1', 'P_PPO_decreasing_2', 'P_PPO_decreasing_3', 'P_PPO_decreasing_4', 'P_PPO_decreasing_5', 'P_PPO_decreasing_6', 'P_PPO_decreasing_7', 'P_PPO_decreasing_8', 'P_PPO_decreasing_9', 'P_PPO_decreasing_10', 'P_PPO_decreasing_11', 'P_PPO_decreasing_12', 'P_PPO_decreasing_13', 'P_PPO_decreasing_14', 'P_PPO_decreasing_15', 'P_PPO_decreasing_16', 'P_PPO_decreasing_17', 'P_PPO_decreasing_18', 'P_PPO_decreasing_19', 'P_PPO_decreasing_20', 'P_DQN_normal_1', 'P_DQN_normal_2', 'P_DQN_normal_3', 'P_DQN_normal_4', 'P_DQN_normal_5', 'P_DQN_normal_6', 'P_DQN_normal_7', 'P_DQN_normal_8', 'P_DQN_normal_9', 'P_DQN_normal_10', 'P_DQN_normal_11', 'P_DQN_normal_12', 'P_DQN_normal_13', 'P_DQN_normal_14', 'P_DQN_normal_15', 'P_DQN_normal_16', 'P_DQN_normal_17', 'P_DQN_normal_18', 'P_DQN_normal_19', 'P_DQN_normal_20', 'P_DQN_increasing_1', 'P_DQN_increasing_2', 'P_DQN_increasing_3', 'P_DQN_increasing_4', 'P_DQN_increasing_5', 'P_DQN_increasing_6', 'P_DQN_increasing_7', 'P_DQN_increasing_8', 'P_DQN_increasing_9', 'P_DQN_increasing_10', 'P_DQN_increasing_11', 'P_DQN_increasing_12', 'P_DQN_increasing_13', 'P_DQN_increasing_14', 'P_DQN_increasing_15', 'P_DQN_increasing_16', 'P_DQN_increasing_17', 'P_DQN_increasing_18', 'P_DQN_increasing_19', 'P_DQN_increasing_20', 'P_DQN_decreasing_1', 'P_DQN_decreasing_2', 'P_DQN_decreasing_3', 'P_DQN_decreasing_4', 'P_DQN_decreasing_5', 'P_DQN_decreasing_6', 'P_DQN_decreasing_7', 'P_DQN_decreasing_8', 'P_DQN_decreasing_9', 'P_DQN_decreasing_10', 'P_DQN_decreasing_11', 'P_DQN_decreasing_12', 'P_DQN_decreasing_13', 'P_DQN_decreasing_14', 'P_DQN_decreasing_15', 'P_DQN_decreasing_16', 'P_DQN_decreasing_17', 'P_DQN_decreasing_18', 'P_DQN_decreasing_19', 'P_DQN_decreasing_20', 'P_A2C_normal_1', 'P_A2C_normal_2', 'P_A2C_normal_3', 'P_A2C_normal_4', 'P_A2C_normal_5', 'P_A2C_normal_6', 'P_A2C_normal_7', 'P_A2C_normal_8', 'P_A2C_normal_9', 'P_A2C_normal_10', 'P_A2C_normal_11', 'P_A2C_normal_12', 'P_A2C_normal_13', 'P_A2C_normal_14', 'P_A2C_normal_15', 'P_A2C_normal_16', 'P_A2C_normal_17', 'P_A2C_normal_18', 'P_A2C_normal_19', 'P_A2C_normal_20', 'P_A2C_increasing_1', 'P_A2C_increasing_2', 'P_A2C_increasing_3', 'P_A2C_increasing_4', 'P_A2C_increasing_5', 'P_A2C_increasing_6', 'P_A2C_increasing_7', 'P_A2C_increasing_8', 'P_A2C_increasing_9', 'P_A2C_increasing_10', 'P_A2C_increasing_11', 'P_A2C_increasing_12', 'P_A2C_increasing_13', 'P_A2C_increasing_14', 'P_A2C_increasing_15', 'P_A2C_increasing_16', 'P_A2C_increasing_17', 'P_A2C_increasing_18', 'P_A2C_increasing_19', 'P_A2C_increasing_20', 'P_A2C_decreasing_1', 'P_A2C_decreasing_2', 'P_A2C_decreasing_3', 'P_A2C_decreasing_4', 'P_A2C_decreasing_5', 'P_A2C_decreasing_6', 'P_A2C_decreasing_7', 'P_A2C_decreasing_8', 'P_A2C_decreasing_9', 'P_A2C_decreasing_10', 'P_A2C_decreasing_11', 'P_A2C_decreasing_12', 'P_A2C_decreasing_13', 'P_A2C_decreasing_14', 'P_A2C_decreasing_15', 'P_A2C_decreasing_16', 'P_A2C_decreasing_17', 'P_A2C_decreasing_18', 'P_A2C_decreasing_19', 'P_A2C_decreasing_20', 'M_PPO_normal_1', 'M_PPO_normal_2', 'M_PPO_normal_3', 'M_PPO_normal_4', 'M_PPO_normal_5', 'M_PPO_normal_6', 'M_PPO_normal_7', 'M_PPO_normal_8', 'M_PPO_normal_9', 'M_PPO_normal_10', 'M_PPO_normal_11', 'M_PPO_normal_12', 'M_PPO_normal_13', 'M_PPO_normal_14', 'M_PPO_normal_15', 'M_PPO_normal_16', 'M_PPO_normal_17', 'M_PPO_normal_18', 'M_PPO_normal_19', 'M_PPO_normal_20', 'M_PPO_increasing_1', 'M_PPO_increasing_2', 'M_PPO_increasing_3', 'M_PPO_increasing_4', 'M_PPO_increasing_5', 'M_PPO_increasing_6', 'M_PPO_increasing_7', 'M_PPO_increasing_8', 'M_PPO_increasing_9', 'M_PPO_increasing_10', 'M_PPO_increasing_11', 'M_PPO_increasing_12', 'M_PPO_increasing_13', 'M_PPO_increasing_14', 'M_PPO_increasing_15', 'M_PPO_increasing_16', 'M_PPO_increasing_17', 'M_PPO_increasing_18', 'M_PPO_increasing_19', 'M_PPO_increasing_20', 'M_PPO_decreasing_1', 'M_PPO_decreasing_2', 'M_PPO_decreasing_3', 'M_PPO_decreasing_4', 'M_PPO_decreasing_5', 'M_PPO_decreasing_6', 'M_PPO_decreasing_7', 'M_PPO_decreasing_8', 'M_PPO_decreasing_9', 'M_PPO_decreasing_10', 'M_PPO_decreasing_11', 'M_PPO_decreasing_12', 'M_PPO_decreasing_13', 'M_PPO_decreasing_14', 'M_PPO_decreasing_15', 'M_PPO_decreasing_16', 'M_PPO_decreasing_17', 'M_PPO_decreasing_18', 'M_PPO_decreasing_19', 'M_PPO_decreasing_20', 'M_DQN_normal_1', 'M_DQN_normal_2', 'M_DQN_normal_3', 'M_DQN_normal_4', 'M_DQN_normal_5', 'M_DQN_normal_6', 'M_DQN_normal_7', 'M_DQN_normal_8', 'M_DQN_normal_9', 'M_DQN_normal_10', 'M_DQN_normal_11', 'M_DQN_normal_12', 'M_DQN_normal_13', 'M_DQN_normal_14', 'M_DQN_normal_15', 'M_DQN_normal_16', 'M_DQN_normal_17', 'M_DQN_normal_18', 'M_DQN_normal_19', 'M_DQN_normal_20', 'M_DQN_increasing_1', 'M_DQN_increasing_2', 'M_DQN_increasing_3', 'M_DQN_increasing_4', 'M_DQN_increasing_5', 'M_DQN_increasing_6', 'M_DQN_increasing_7', 'M_DQN_increasing_8', 'M_DQN_increasing_9', 'M_DQN_increasing_10', 'M_DQN_increasing_11', 'M_DQN_increasing_12', 'M_DQN_increasing_13', 'M_DQN_increasing_14', 'M_DQN_increasing_15', 'M_DQN_increasing_16', 'M_DQN_increasing_17', 'M_DQN_increasing_18', 'M_DQN_increasing_19', 'M_DQN_increasing_20', 'M_DQN_decreasing_1', 'M_DQN_decreasing_2', 'M_DQN_decreasing_3', 'M_DQN_decreasing_4', 'M_DQN_decreasing_5', 'M_DQN_decreasing_6', 'M_DQN_decreasing_7', 'M_DQN_decreasing_8', 'M_DQN_decreasing_9', 'M_DQN_decreasing_10', 'M_DQN_decreasing_11', 'M_DQN_decreasing_12', 'M_DQN_decreasing_13', 'M_DQN_decreasing_14', 'M_DQN_decreasing_15', 'M_DQN_decreasing_16', 'M_DQN_decreasing_17', 'M_DQN_decreasing_18', 'M_DQN_decreasing_19', 'M_DQN_decreasing_20', 'M_A2C_normal_1', 'M_A2C_normal_2', 'M_A2C_normal_3', 'M_A2C_normal_4', 'M_A2C_normal_5', 'M_A2C_normal_6', 'M_A2C_normal_7', 'M_A2C_normal_8', 'M_A2C_normal_9', 'M_A2C_normal_10', 'M_A2C_normal_11', 'M_A2C_normal_12', 'M_A2C_normal_13', 'M_A2C_normal_14', 'M_A2C_normal_15', 'M_A2C_normal_16', 'M_A2C_normal_17', 'M_A2C_normal_18', 'M_A2C_normal_19', 'M_A2C_normal_20', 'M_A2C_increasing_1', 'M_A2C_increasing_2', 'M_A2C_increasing_3', 'M_A2C_increasing_4', 'M_A2C_increasing_5', 'M_A2C_increasing_6', 'M_A2C_increasing_7', 'M_A2C_increasing_8', 'M_A2C_increasing_9', 'M_A2C_increasing_10', 'M_A2C_increasing_11', 'M_A2C_increasing_12', 'M_A2C_increasing_13', 'M_A2C_increasing_14', 'M_A2C_increasing_15', 'M_A2C_increasing_16', 'M_A2C_increasing_17', 'M_A2C_increasing_18', 'M_A2C_increasing_19', 'M_A2C_increasing_20', 'M_A2C_decreasing_1', 'M_A2C_decreasing_2', 'M_A2C_decreasing_3', 'M_A2C_decreasing_4', 'M_A2C_decreasing_5', 'M_A2C_decreasing_6', 'M_A2C_decreasing_7', 'M_A2C_decreasing_8', 'M_A2C_decreasing_9', 'M_A2C_decreasing_10', 'M_A2C_decreasing_11', 'M_A2C_decreasing_12', 'M_A2C_decreasing_13', 'M_A2C_decreasing_14', 'M_A2C_decreasing_15', 'M_A2C_decreasing_16', 'M_A2C_decreasing_17', 'M_A2C_decreasing_18', 'M_A2C_decreasing_19', 'M_A2C_decreasing_20', 'G_PPO_normal_1', 'G_PPO_normal_2', 'G_PPO_normal_3', 'G_PPO_normal_4', 'G_PPO_normal_5', 'G_PPO_normal_6', 'G_PPO_normal_7', 'G_PPO_normal_8', 'G_PPO_normal_9', 'G_PPO_normal_10', 'G_PPO_normal_11', 'G_PPO_normal_12', 'G_PPO_normal_13', 'G_PPO_normal_14', 'G_PPO_normal_15', 'G_PPO_normal_16', 'G_PPO_normal_17', 'G_PPO_normal_18', 'G_PPO_normal_19', 'G_PPO_normal_20', 'G_PPO_increasing_1', 'G_PPO_increasing_2', 'G_PPO_increasing_3', 'G_PPO_increasing_4', 'G_PPO_increasing_5', 'G_PPO_increasing_6', 'G_PPO_increasing_7', 'G_PPO_increasing_8', 'G_PPO_increasing_9', 'G_PPO_increasing_10', 'G_PPO_increasing_11', 'G_PPO_increasing_12', 'G_PPO_increasing_13', 'G_PPO_increasing_14', 'G_PPO_increasing_15', 'G_PPO_increasing_16', 'G_PPO_increasing_17', 'G_PPO_increasing_18', 'G_PPO_increasing_19', 'G_PPO_increasing_20', 'G_PPO_decreasing_1', 'G_PPO_decreasing_2', 'G_PPO_decreasing_3', 'G_PPO_decreasing_4', 'G_PPO_decreasing_5', 'G_PPO_decreasing_6', 'G_PPO_decreasing_7', 'G_PPO_decreasing_8', 'G_PPO_decreasing_9', 'G_PPO_decreasing_10', 'G_PPO_decreasing_11', 'G_PPO_decreasing_12', 'G_PPO_decreasing_13', 'G_PPO_decreasing_14', 'G_PPO_decreasing_15', 'G_PPO_decreasing_16', 'G_PPO_decreasing_17', 'G_PPO_decreasing_18', 'G_PPO_decreasing_19', 'G_PPO_decreasing_20', 'G_DQN_normal_1', 'G_DQN_normal_2', 'G_DQN_normal_3', 'G_DQN_normal_4', 'G_DQN_normal_5', 'G_DQN_normal_6', 'G_DQN_normal_7', 'G_DQN_normal_8', 'G_DQN_normal_9', 'G_DQN_normal_10', 'G_DQN_normal_11', 'G_DQN_normal_12', 'G_DQN_normal_13', 'G_DQN_normal_14', 'G_DQN_normal_15', 'G_DQN_normal_16', 'G_DQN_normal_17', 'G_DQN_normal_18', 'G_DQN_normal_19', 'G_DQN_normal_20', 'G_DQN_increasing_1', 'G_DQN_increasing_2', 'G_DQN_increasing_3', 'G_DQN_increasing_4', 'G_DQN_increasing_5', 'G_DQN_increasing_6', 'G_DQN_increasing_7', 'G_DQN_increasing_8', 'G_DQN_increasing_9', 'G_DQN_increasing_10', 'G_DQN_increasing_11', 'G_DQN_increasing_12', 'G_DQN_increasing_13', 'G_DQN_increasing_14', 'G_DQN_increasing_15', 'G_DQN_increasing_16', 'G_DQN_increasing_17', 'G_DQN_increasing_18', 'G_DQN_increasing_19', 'G_DQN_increasing_20', 'G_DQN_decreasing_1', 'G_DQN_decreasing_2', 'G_DQN_decreasing_3', 'G_DQN_decreasing_4', 'G_DQN_decreasing_5', 'G_DQN_decreasing_6', 'G_DQN_decreasing_7', 'G_DQN_decreasing_8', 'G_DQN_decreasing_9', 'G_DQN_decreasing_10', 'G_DQN_decreasing_11', 'G_DQN_decreasing_12', 'G_DQN_decreasing_13', 'G_DQN_decreasing_14', 'G_DQN_decreasing_15', 'G_DQN_decreasing_16', 'G_DQN_decreasing_17', 'G_DQN_decreasing_18', 'G_DQN_decreasing_19', 'G_DQN_decreasing_20', 'G_A2C_normal_1', 'G_A2C_normal_2', 'G_A2C_normal_3', 'G_A2C_normal_4', 'G_A2C_normal_5', 'G_A2C_normal_6', 'G_A2C_normal_7', 'G_A2C_normal_8', 'G_A2C_normal_9', 'G_A2C_normal_10', 'G_A2C_normal_11', 'G_A2C_normal_12', 'G_A2C_normal_13', 'G_A2C_normal_14', 'G_A2C_normal_15', 'G_A2C_normal_16', 'G_A2C_normal_17', 'G_A2C_normal_18', 'G_A2C_normal_19', 'G_A2C_normal_20', 'G_A2C_increasing_1', 'G_A2C_increasing_2', 'G_A2C_increasing_3', 'G_A2C_increasing_4', 'G_A2C_increasing_5', 'G_A2C_increasing_6', 'G_A2C_increasing_7', 'G_A2C_increasing_8', 'G_A2C_increasing_9', 'G_A2C_increasing_10', 'G_A2C_increasing_11', 'G_A2C_increasing_12', 'G_A2C_increasing_13', 'G_A2C_increasing_14', 'G_A2C_increasing_15', 'G_A2C_increasing_16', 'G_A2C_increasing_17', 'G_A2C_increasing_18', 'G_A2C_increasing_19', 'G_A2C_increasing_20', 'G_A2C_decreasing_1', 'G_A2C_decreasing_2', 'G_A2C_decreasing_3', 'G_A2C_decreasing_4', 'G_A2C_decreasing_5', 'G_A2C_decreasing_6', 'G_A2C_decreasing_7', 'G_A2C_decreasing_8', 'G_A2C_decreasing_9', 'G_A2C_decreasing_10', 'G_A2C_decreasing_11', 'G_A2C_decreasing_12', 'G_A2C_decreasing_13', 'G_A2C_decreasing_14', 'G_A2C_decreasing_15', 'G_A2C_decreasing_16', 'G_A2C_decreasing_17', 'G_A2C_decreasing_18', 'G_A2C_decreasing_19', 'G_A2C_decreasing_20']


def lanzarExperimento(nombreExp):
    models_dir = "modelsPaper/" + nombreExp
    log_dir = "logsPaper/"

    caso = nombreExp.split("_")

    if caso[0] == 'P':
        nVehiculos = 5
        nNodos = 20
    
    if caso[0] == 'M':
        nVehiculos = 13
        nNodos = 50

    if caso[0] == 'G':
        nVehiculos = 25
        nNodos = 100

    crearDirectorios(models_dir, log_dir)
    
    env = crearEnv(nVehiculos, nNodos, caso[2])

    model = crearModelo(caso[1], env, log_dir)
    
    eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                            eval_freq=40000, deterministic=True, render=False)

    new_logger = configure(format_strings = ["log", "csv", "tensorboard"])

    start_time = time.time()


    for _ in range(1, ITERATIONS+1):
        model.set_logger(new_logger)
        model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = nombreExp, callback=eval_callback)

    print("---%s: %s minutos ---" % (nombreExp, round((time.time() - start_time)/60, 2)))

    env.close()

if __name__ == '__main__':
    pool = Pool(processes = 4)
    result = pool.map(lanzarExperimento, (listaPruebas))
    pool.close()
    pool.join()
    
"""
Nota: usar el comando python experimentacion.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""
