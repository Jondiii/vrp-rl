from stable_baselines3 import PPO, A2C
from vrpEnv import VRPEnv
import os
import time

"""
Definimos primero dónde buscar el modelo ya entrenado.
"""

model_name = "PPO_increasing"
models_dir = "logsPaper2" # Sin el -1 de las acciones, no funciona ni tan mal, pero tarda la vida. 
model_path = f"{models_dir}/{model_name}"

"""
INICIALIZACIÓN DE ENTORNO Y AGENTE
"""
env = VRPEnv()
env.readEnvFromFile(7, 20, 13, 50, 'data/case20n7v')
#env.setIncreasingIsDone(5000)
env.reset()

model = PPO.load(model_path, env)

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