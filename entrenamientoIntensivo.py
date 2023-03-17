from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, DQN
from dataGenerator import  DataGenerator
from vrpEnv import VRPEnv
import os
import time


numCasos = 10
numNodos = 50
numVehiculos = 20

n_maxDemand = 30
n_twMin = None
n_twMax = None

v_maxDemads = 100
v_speed = 70


dataFolder = "data/intensivo"

for i in range(numCasos):
    dataPath = os.path.join(dataFolder, "case" + str(i+1))
    dataGen = DataGenerator(numNodos, numVehiculos, dataPath, seed = i) # La i es la semilla
    
    dataGen.addNodeInfo(n_maxDemand, n_twMin, n_twMax)
    dataGen.addVehicleInfo(v_maxDemads, v_speed)
    dataGen.generateNodeInfo()
    dataGen.generateVehicleInfo()
    dataGen.saveData()

"""

ALGORTIHM = "PPO"

model_name = "100000"

models_dir = "models/" + ALGORTIHM



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
"""