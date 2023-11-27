from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, DQN, A2C
from vrpEnv import VRPEnv
import os
import time
import numpy as np

ITERATIONS = 5
TIMESTEPS = 2048*100 # Serán 5M de steps

listaMetodo = ["normal", "increasing", "decreasing"]
listaAlgoritmo = ["PPO", "DQN", "A2C"]
#listaTamanyo = ["P", "M", "G"]
#listaExpNumber = [*range(20)]

nVehiculos = 13
nNodos = 50

class CustomCallback(BaseCallback):
    shortestRoute = 100000000000000
    shortestRouteNodesVisited = None

    def __init__(self, verbose=0):
            super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        env = self.model.get_env()
        tiempoTotal = 0
        tiempos = env.get_attr("currTime")

        for tiempo in tiempos[0]:
                tiempoTotal += tiempo

        print(tiempoTotal)
        print(self.shortestRoute)

        if tiempoTotal < self.shortestRoute:
            self.shortestRoute = tiempoTotal

            self.shortestRouteNodesVisited = np.count_nonzero(env.get_attr("visited")[:env.get_attr("nNodos")[0]] == 1) / env.get_attr("nNodos")[0]

            with open("resultsPaper/"+env.get_attr("name")[0]+".txt", 'w', encoding="utf-8") as f:
                f.write(env.get_attr("name")[0])
                f.write("\n")
                f.write(str(self.shortestRoute))
                f.write("\n")
                f.write(str(self.shortestRouteNodesVisited)+"%")
                f.write("\n")

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


def crearEnv(nVehiculos, nNodos, metodo, name):
    env = VRPEnv(multiTrip = True, name = name) 
    env.createEnv(nVehiculos = nVehiculos, nNodos = nNodos, maxNodeCapacity = 4, sameMaxNodeVehicles=True)

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

        callback = CustomCallback()

        crearDirectorios(models_dir, log_dir, result_dir)

        env = crearEnv(nVehiculos, nNodos, metodo, name =  nombreExp)

        model = crearModelo(algoritmo, env, log_dir)

        start_time = time.time()

        for i in range(1, ITERATIONS+1):
            model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = nombreExp, callback=callback)

        model.save(f"{models_dir}/final")

        print("---%s: %s minutos ---" % (nombreExp, round((time.time() - start_time)/60, 2)))

        env.render("resultsPaper/"+nombreExp )
        env.close()


"""
Nota: usar el comando python experimentacion.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""
