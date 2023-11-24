from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, DQN, A2C
from vrpEnv import VRPEnv
import os
import time
import numpy as np

ITERATIONS = 500
TIMESTEPS = 2048*10 # Serán 5M de steps

listaMetodo = ["normal", "increasing", "decreasing"]
listaAlgoritmo = ["PPO", "DQN", "A2C"]
#listaTamanyo = ["P", "M", "G"]
#listaExpNumber = [*range(20)]

nVehiculos = 13
nNodos = 50

class CustomCallback(BaseCallback):
    shortestRoute = 0
    shortestRouteNodesVisited = None

    def _on_rollout_end(self) -> None:
        env = self.model.get_env()
        
        tiempoTotal = 0
        for _, tiempo in zip(env.v_ordenVisitas, env.currTime):
                tiempoTotal += tiempo

        if tiempoTotal < shortestRoute:
            shortestRoute = tiempoTotal

            shortestRouteNodesVisited = np.count_nonzero(env.visited[:env.nNodos] == 1) / env.nNodos
            
            with open("resultsPaper/"+env.name+".txt", 'w', encoding="utf-8") as f:
                f.write(env.name)
                f.write(str(shortestRoute))
                f.write(str(shortestRouteNodesVisited)+"%")

                f.close()


def crearDirectorios(models_dir, log_dir, result_dir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

def crearModelo(algoritmo, env, log_dir):
    if algoritmo == 'PPO':
        return PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    if algoritmo == 'A2C':
        return  A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    if algoritmo == 'DQN':
        return DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")


def crearEnv(nVehiculos, nNodos, metodo, nombreExp):
    env = VRPEnv(multiTrip = True) 
    env.createEnv(nVehiculos = nVehiculos, nNodos = nNodos, maxNodeCapacity = 4, sameMaxNodeVehicles=True, name = nombreExp)

    if metodo == "decreasing":
        env.setDecayingIsDone(ITERATIONS * TIMESTEPS)

    if metodo == "increasing":
        env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)

    env.reset()

    return env


for algoritmo in listaAlgoritmo:
    for metodo in listaMetodo:
        nombreExp = algoritmo + "_" + metodo

        models_dir = "modelsPaper/" + nombreExp
        log_dir = "logsPaper"
        result_dir = "resultsPaper"

        crearDirectorios(models_dir, log_dir, result_dir)

        env = crearEnv(nVehiculos, nNodos, metodo, name =  nombreExp)

        model = crearModelo(algoritmo, env, log_dir)

        start_time = time.time()

        for i in range(1, ITERATIONS+1):
            model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = algoritmo, callback=CustomCallback())

        model.save(f"{models_dir}/final")
        env.render(nombreExp)

        print("---%s: %s minutos ---" % (nombreExp, round((time.time() - start_time)/60, 2)))


        env.render("renderPaper/"+nombreExp )
        env.close()


"""
Nota: usar el comando python experimentacion.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""
