from stable_baselines3 import PPO
from vrpEnv import VRPEnv
import os
import time

ALGORTIHM = "PPO"

model_name = "model"

#models_dir = "models/" + ALGORTIHM
models_dir = "Resultados/Prueba_006"
model_path = f"{models_dir}/{model_name}"

env = VRPEnv()
env.createEnv(nVehiculos = 10, nNodos = 100, maxNodeCapacity = 2, maxNumVehiculos = 10, maxNumNodos = 100,)
#env.setIncreasingIsDone(200)
env.reset()

model = PPO.load(model_path, env)

episodes = 1

start_time = time.time()

for ep in range(episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    env.graphicalRender()

    env.close()

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))