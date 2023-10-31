from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN, A2C
from vrpEnv import VRPEnv
import os
import time

ITERATIONS = 500
TIMESTEPS = 2048*10 # Serán 5M de steps

listaMetodo = ["normal", "increasing", "decreasing"]
listaTamanyo = ["P", "M", "G"]
listaExpNumber = [*range(20)]
algoritmo = 'PPO'


def crearDirectorios(models_dir, log_dir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def crearModelo(env, log_dir):
    return PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")


def crearEnv(nVehiculos, nNodos, metodo):
    env = VRPEnv(multiTrip = True) #TODO Con esto las rutas que salen al final son de vehiculos yendo y viniendo del depot con un solo pedido
    env.createEnv(nVehiculos = nVehiculos, nNodos = nNodos, maxNodeCapacity = 4, sameMaxNodeVehicles=True)

    if metodo == "decreasing":
        env.setDecayingIsDone(ITERATIONS * TIMESTEPS)

    if metodo == "increasing":
        env.setIncreasingIsDone(ITERATIONS * TIMESTEPS)

    env.reset()

    return env


for tamanyo in listaTamanyo:
    if tamanyo == 'P':
        nVehiculos = 5
        nNodos = 20
    
    if tamanyo == 'M':
        nVehiculos = 13
        nNodos = 50

    if tamanyo == 'G':
        nVehiculos = 25
        nNodos = 100

    for metodo in listaMetodo:
        for expNum in listaExpNumber:
            nombreExp = tamanyo + "_" + algoritmo + "_" + metodo + "_" + expNum

            models_dir = "modelsPaper/" + nombreExp
            log_dir = "logsPaper"

            crearDirectorios(models_dir, log_dir)

            env = crearEnv(nVehiculos, nNodos, metodo)

            model = crearModelo(env, log_dir)

            start_time = time.time()

            for i in range(1, ITERATIONS+1):
                model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = algoritmo)

            model.save(f"{models_dir}/final")

            print("---%s: %s minutos ---" % (nombreExp, round((time.time() - start_time)/60, 2)))


            env.render("renderPaper/"+nombreExp )
            env.close()


    """
    Nota: usar el comando python experimentacion.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
    """
