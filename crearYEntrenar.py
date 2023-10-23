from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN, A2C
from vrpEnv import VRPEnv
import os
import time

"""
Definimos primero nombres de carpetas, para que se puedan crear en caso de no existir.
"""

ALGORTIHM = "entregaFleetbot_A2C_M_I" # Nombre de la ejecución (no afecta al algoritmo que se vaya a usar)
models_dir = "modelsFleetbot/" + ALGORTIHM # Directorio donde guardar los modelos generados
log_dir = "logsFleetbot_A2C_M_I"          # Directorios donde guardar los logs

ITERATIONS = 100          # Número de iteraciones
TIMESTEPS = 2048*10       # Pasos por cada iteración (poner múltiplos de 2048)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


"""
INICIALIZACIÓN DE ENTORNO Y AGENTE
"""
env = VRPEnv()  # Creamos un entorno vacío
env.createEnv(nVehiculos = 13, nNodos = 50, maxNodeCapacity = 4, sameMaxNodeVehicles=True)
env.reset()     # Siempre hay que resetear el entorno nada más crearlo

# Creamos el modelo. Se puede usar un algoritmo u otro simplemente cambiando el constructor
# al correspondiente. Lo que hay dentro del constructor no hace falta cambiarlo.
model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device = "cuda")

start_time = time.time()

"""
ENTRENAMIENTO
"""
for i in range(1, ITERATIONS+1):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

env.render(ALGORTIHM) # Crea gráficas y un report con las últimas rutas generadas. Recibe el nombre de la carpeta donde se guarda

env.close()


"""
Nota: usar el comando python crearYEntrenar.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
"""
