from stable_baselines3 import PPO, A2C
from vrpEnv import VRPEnv
import os
import time

"""
Definimos primero dónde buscar el modelo ya entrenado.
"""

model_name = "20480000.zip"
models_dir = "modelsFleetbot\entregaFleetbot_A2C_M_I_2" # Sin el -1 de las acciones, no funciona ni tan mal, pero tarda la vida. 
model_path = f"{models_dir}/{model_name}"

"""
INICIALIZACIÓN DE ENTORNO Y AGENTE
"""
env = VRPEnv()
env.createEnv(nVehiculos = 13, nNodos = 50, maxNodeCapacity = 4, sameMaxNodeVehicles=True)
#env.setIncreasingIsDone(200)
env.reset()

model = A2C.load(model_path, env)

# Indicamos el número de episodios (a más episodios más soluciones obtendremos)
episodes = 1

start_time = time.time()

"""
GENERACIÓN DE RUTAS
"""
for ep in range(episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        print(obs["n_visited"])

    env.render("demo2") # Guarda un report y los grafos en la ruta especificada.
    # env.graphicalRender() # TODO
    
    env.close()

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))