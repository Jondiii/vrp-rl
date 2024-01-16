from stable_baselines3 import PPO, A2C, DQN
from vrpEnv import VRPEnv
import os
import time

"""
Definimos primero dónde buscar el modelo ya entrenado.
"""

listaMetodo = ["normal", "increasing", "decreasing"]
listaAlgoritmo = ["PPO"]
listaTamanyo = [(7, 20), (13,50)]

models_dir = 'logsPaper3'
model_extension = '.zip'

start_time = time.time()

for tamanyo in listaTamanyo:
    nVehiculos, nNodos = tamanyo

    dataPath = 'data/case20n7v' if nVehiculos == 7 else 'data/case50n13v'

    for algoritmo in listaAlgoritmo:
        for metodo in listaMetodo:
            expCase = algoritmo + '_' + metodo
            model_path = f"{models_dir}/{metodo}/{expCase + model_extension}"

            env = VRPEnv()
            env.readEnvFromFile(nVehiculos, nNodos, 13, 50, dataPath)
            env.reset()

            if algoritmo == 'PPO':
                model = PPO.load(model_path, env)
            
            if algoritmo == 'DQN':
                model = DQN.load(model_path, env)
            
            if algoritmo == 'A2C':
                model = A2C.load(model_path, env)

            episodes = 1

            for ep in range(episodes):
                obs = env.reset()
                done = False
                
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, info = env.step(action)

                env.render('grafosExpFin2/' + expCase + "_" + str(nNodos)) # Guarda un report y los grafos en la ruta especificada.
            
                env.close()

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))