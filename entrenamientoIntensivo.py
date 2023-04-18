from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, DQN
from dataGenerator import  DataGenerator
from vrpEnv import VRPEnv
import os
import time

"""
Este script crea tantos casos de uso como numCasos y los guarda en dataFolder. A continuación, realiza un entrenamiento intensivo en el que
se usan todos estos casos de forma secuencial. Se realiza un entrenamiento inicial con el primer caso durante TIMESTEPS, tras lo que se pasa
a entrenar con el siguiente caso, de nuevo durante TIMESTEPS. Esto se hace tantas veces como numCasos, y una vez llegado al último se
vuelve a empezar desde el primero. Este proceso se repite ITERATIONS veces, por lo que el total de timesteps final del entrenamiento sería
TIMESTEPS * numCasos * TIMESTEPS
"""

ITERATIONS = 20
TIMESTEPS = 2048*10 # Poner múltiplos de 2048
numCasos = 10

numNodos = 50
numVehiculos = 20

n_maxDemand = 30
n_twMin = None
n_twMax = None

v_maxDemads = 100
v_speed = 70


dataFolder = "data/intensivo"

ALGORTIHM = "PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


for i in range(numCasos):
    dataPath = os.path.join(dataFolder, "case" + str(i+1))
    dataGen = DataGenerator(numNodos + 1, numVehiculos, dataPath, seed = i) # La i es la semilla
    
    dataGen.addNodeInfo(n_maxDemand, n_twMin, n_twMax)
    dataGen.addVehicleInfo(v_maxDemads, v_speed)
    dataGen.generateNodeInfo()
    dataGen.generateVehicleInfo()
    dataGen.saveData()


# Como se va a entrenar un modelo con múltiples variantes de un mismo entorno, primero hay que crear el modelo y entrenarlo un poco
# Una vez creado, en las siguientes iteraciones se cargará dicho modelo, para entrenarlo con un entorno algo distinto.
env = VRPEnv(multiTrip = True)
env.readEnvFromFile(numVehiculos, numNodos, filePath=os.path.join(dataFolder, "case1"))
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

start_time = time.time()


model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = ALGORTIHM)
model.save(f"{models_dir}/{TIMESTEPS}")

for j in range(1, ITERATIONS + 1):
    for i in range(1, numCasos + 1):
        env.close()

        env = VRPEnv(multiTrip = True)
        env.readEnvFromFile(numVehiculos, numNodos, filePath=os.path.join(dataFolder, "case" + i))
        env.reset()

        model = PPO.load(f"{models_dir}/{TIMESTEPS * i * j}", env)

        model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = ALGORTIHM)

        model.save(f"{models_dir}/{TIMESTEPS * (j*i+1)}")

        env.render()


print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

env.close()
