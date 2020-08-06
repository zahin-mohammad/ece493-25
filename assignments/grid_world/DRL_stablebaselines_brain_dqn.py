from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
import numpy as np
from custom_gym import CustomGym

tasks = []
# Task 1
tasks.append((np.array([[2, 2], [3, 6]]), np.array([[6, 3], [1, 4]])))

# Task 2
tasks.append((np.array([[6, 2], [5, 2], [4, 2], [3, 2], [2, 2], [6, 3], [6, 4], [6, 5],
                        [2, 3], [2, 4], [2, 5]]), []))
# Task 3
tasks.append((np.array([[6, 3], [6, 3], [6, 2], [5, 2], [4, 2], [3, 2], [3, 3],
                        [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [5, 7], [7, 3]]), np.array([[1, 3], [0, 5], [7, 7], [8, 5]])))

experiments = {}
agentXY = [0, 0]
goalXY = [4, 4]

# showRender = True
episodes = 2000
# renderEveryNth = 10
# printEveryNth = 1
# do_plot_rewards = True

env = Monitor(CustomGym(agentXY, goalXY, tasks[0][0], tasks[0][1], title=f"Deep Q {task+1}"), filename=None)
model = DQN(MlpPolicy, env, verbose=1)

while len(env.get_episode_rewards()) < episodes:
    model.learn(total_timesteps=25000)



# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()