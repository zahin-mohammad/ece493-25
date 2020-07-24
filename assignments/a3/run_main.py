from maze_env import Maze
from RL_brainsample_PI import rlalgorithm as rlalg1
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def plot_rewards(experiments):
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    for i, (env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    plt.show()

def update(env, RL, data, episodes=50):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward

    for episode in range(episodes):
        t=0
        # initial state
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            if(showRender or (episode % renderEveryNth)==0):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))

            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    # end of game
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    env.destroy()

if __name__ == "__main__":
    sim_speed = 0.05

    #Example Short Fast for Debugging
    showRender=True
    episodes=30
    renderEveryNth=5
    printEveryNth=1
    do_plot_rewards=True

    #Example Full Run, you may need to run longer
    #showRender=False
    #episodes=2000
    #renderEveryNth=10000
    #printEveryNth=100
    #do_plot_rewards=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    # Task Specifications

    agentXY=[0,0]
    goalXY=[4,4]

    # Task 1
    wall_shape=np.array([[2,2],[3,6]])
    pits=np.array([[6,3],[1,4]])

    # Task 2
    wall_shape=np.array([[6,2],[5,2],[4,2],[3,2],[2,2],[6,3],[6,4],[6,5],
        [2,3],[2,4],[2,5]])
    pits=[]

    # Task 3
    wall_shape=np.array([[6,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],
        [3,4],[3,5],[3,6],[4,6],[5,6],[5,7],[7,3]])
    pits=np.array([[1,3],[0,5], [7,7], [8,5]])

    env1 = Maze(agentXY,goalXY,wall_shape, pits)
    RL1 = rlalg1(actions=list(range(env1.n_actions)))
    data1={}
    env1.after(10, update(env1, RL1, data1, episodes))
    env1.mainloop()
    experiments = [(env1,RL1, data1)]

    #Create another RL_brain_ALGNAME.py class and import it as rlag2 then run it here.
    #env2 = Maze(agentXY,goalXY,wall_shape,pits)
    #RL2 = rlalg2(actions=list(range(env2.n_actions)))
    #data2={}
    #env2.after(10, update(env2, RL2, data2, episodes))
    #env2.mainloop()
    #experiments.append((env2,RL2, data2))


    print("All experiments complete")

    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))


    if(do_plot_rewards):
        #Simple plot of return for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments)

    #Not implemented yet
    #if(do_save_data):
    #    for env, RL, data in experiments:
    #        saveData(env,RL,data)

