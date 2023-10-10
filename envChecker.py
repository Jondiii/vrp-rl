from vrpEnv import VRPEnv

from stable_baselines3.common.env_checker import check_env

"""
Stable Baselines 3 contiene una variedad de entornos listos para usar, pero también permite crear entornos propios.
El método env_checker sirve para comprobar que los entornos creados por los usuarios sean consistentes en cuanto
al espacio de acciones y de observaciones, mirando que los espacios declarados coinciden con las observaciones
que devuelve el entorno en cada iteración.

Que un entorno pase el checker no quiere decir que esté libre de errores.
"""

# Se crea un entorno de prueba.
env = VRPEnv()
env.createEnv(50, 100, sameMaxNodeVehicles=True)


# Se comprueba que el entorno sea consistente.
check_env(env)
