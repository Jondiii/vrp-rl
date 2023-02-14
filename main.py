from vrpEnv import VRPEnv
#from vrpAgent import VRPAgent
import pandas as pd

from stable_baselines import PPO

#dataFleet = pd.read_excel('G:\\.shortcut-targets-by-id\\1qK9SxHRVCvkj4iS9W-ahASG8b_uAaBAV\\2022 11 FLEETBOT\\PruebasRL\\vrpgym\\VRP-GYM\\datos\\Caso100Jobs.xlsb', skiprows=1)

#TODO: meterle los datos de un excel
#print(dataFleet.loc[0])

# Init the environment
env = VRPEnv(nNodos=5, nGrafos=6)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

# Init the agent
#agent = VRPAgent()

# Start training
#agent.train(env, epochs = 10)

# El agente principal es graph_tsp_agent.py, los cambios a la lógica van ahí.