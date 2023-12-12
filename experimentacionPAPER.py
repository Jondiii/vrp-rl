from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN, A2C
from vrpEnv import VRPEnv
import os
import time
import numpy as np

ITERATIONS = 30
TIMESTEPS = 2048*100 # Serán 5M de steps

listaMetodo = ["normal", "increasing", "decreasing"]
listaAlgoritmo = ["PPO"]
#listaAlgoritmo = ["PPO", "DQN", "A2C"]
#listaTamanyo = ["P", "M", "G"]
#listaExpNumber = [*range(20)]

nVehiculos = 13
nNodos = 50

class CustomCallback(BaseCallback):
    shortestRoute = 100000000000000
    shortestRouteNodesVisited = None
    firstTime = False
    
    def __init__(self, check_freq, log_dir, bestModelName, verbose=0):
            super().__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.bestModelName = bestModelName
            self.save_path = os.path.join(log_dir, bestModelName)
            self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)

        return True


    def _on_rollout_start(self) -> None:
        env = self.model.get_env()
        tiempoTotal = 0
        tiempos = env.get_attr("currTime")

        for tiempo in tiempos[0]:
                tiempoTotal += tiempo

        if tiempoTotal < self.shortestRoute:
            self.shortestRoute = tiempoTotal

            self.shortestRouteNodesVisited = np.count_nonzero(env.get_attr("visited")[:env.get_attr("nNodos")[0]] == 1) / env.get_attr("nNodos")[0]

            with open("resultsPaper2/start/"+env.get_attr("name")[0]+".txt", 'w', encoding="utf-8") as f:
                f.write(env.get_attr("name")[0])
                f.write("\n")
                f.write(str(self.shortestRoute))
                f.write("\n")
                f.write(str(self.shortestRouteNodesVisited)+"%")
                f.write("\n")

                f.close()


    def _on_rollout_end(self) -> None:
        env = self.model.get_env()
        tiempoTotal = 0
        tiempos = env.get_attr("currTime")

        for tiempo in tiempos[0]:
                tiempoTotal += tiempo

        if tiempoTotal < self.shortestRoute:
            self.shortestRoute = tiempoTotal

            self.shortestRouteNodesVisited = np.count_nonzero(env.get_attr("visited")[:env.get_attr("nNodos")[0]] == 1) / env.get_attr("nNodos")[0]

            with open("resultsPaper2/"+env.get_attr("name")[0]+".txt", 'w', encoding="utf-8") as f:
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
        os.makedirs(result_dir + "/start")


def crearModelo(algoritmo, env, log_dir):
    if algoritmo == 'PPO':
        return PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    if algoritmo == 'A2C':
        return  A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

    if algoritmo == 'DQN':
        return DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")


def crearEnv(nVehiculos, nNodos, metodo, name):
    env = VRPEnv(multiTrip = False, name = name, seed = 6) 
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

        models_dir = "modelsPaper2/" + nombreExp
        log_dir = "logsPaper2"
        result_dir = "resultsPaper2"

        callback = CustomCallback(500, log_dir, nombreExp)

        crearDirectorios(models_dir, log_dir, result_dir)

        env = crearEnv(nVehiculos, nNodos, metodo, name =  nombreExp)
        env = Monitor(env, log_dir)

        model = crearModelo(algoritmo, env, log_dir)

        start_time = time.time()

        for i in range(1, ITERATIONS+1):
            model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = nombreExp, callback=callback)

        model.save(f"{models_dir}/final")

        print("---%s: %s minutos ---" % (nombreExp, round((time.time() - start_time)/60, 2)))

        env.render("resultsPaper2/"+nombreExp )
        env.close()


"""
Nota: usar el comando python experimentacion.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""
