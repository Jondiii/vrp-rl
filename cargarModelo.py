from stable_baselines3 import PPO
from vrpEnv import VRPEnv
import os
import time

ALGORTIHM = "PPO"

model_name = "100000"

models_dir = "models/" + ALGORTIHM
#models_dir = "Resultados/Prueba_003"
model_path = f"{models_dir}/{model_name}"

env = VRPEnv(nVehiculos = 3, nNodos = 10, maxNumNodos=20, maxNumVehiculos=5)

env.reset()

model = PPO.load(model_path, env)

episodes = 10

start_time = time.time()

for ep in range(episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    
    env.render()
    env.close()

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))