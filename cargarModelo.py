from stable_baselines3 import PPO, A2C
from vrpEnv import VRPEnv
import os
import time
#model_name = "normal_A2C_P.zip" # A2C_I_P mal

model_name = "122880.zip"

models_dir = "models/A2C_Aver"
#models_dir = "modelsEXP"
model_path = f"{models_dir}/{model_name}"

env = VRPEnv()

env.createEnv(nVehiculos = 10, nNodos = 50, maxNodeCapacity = 2, maxNumVehiculos = 10, maxNumNodos = 50)
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

    env.render()
    env.graphicalRender()
    

    env.close()

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))