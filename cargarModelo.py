from stable_baselines3 import PPO, A2C
from vrpEnv import VRPEnv
import os
import time

#model_name = "normal_A2C_P.zip" # A2C_I_P mal

model_name = "6144000.zip"

models_dir = "modelIntenso"
#models_dir = "modelsEXP"

model_path = f"{models_dir}/{model_name}"

#env = VRPEnv(singlePlot = True)
env = VRPEnv()

env.createEnv(nVehiculos = 5, nNodos = 25, maxNodeCapacity = 4, maxNumVehiculos = 5, maxNumNodos = 25)
#env.setIncreasingIsDone(200)
env.reset()

model = A2C.load(model_path, env)

episodes = 1

start_time = time.time()

for ep in range(episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        print(obs["n_visited"])

    env.render(model_name)
    env.graphicalRender()
    

    env.close()

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))