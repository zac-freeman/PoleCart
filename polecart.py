import gym
import math
import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt

def sigmoid(x):
    return ( 1/(1 + np.exp(-x)) )

def makeChoice(outputs):
    return int( outputs[1] > nprnd.random()*(outputs[1] + outputs[0]) )

def setChoices(inputs, weights):
    outputs = [0,0]
    for i in range( len(outputs) ):
        for j in range( len(inputs) ):
            outputs[i] += weights[i][j]*inputs[j]
    return sigmoid(outputs[0]), sigmoid(outputs[1])

def learn(inputList, outputList, actionList, weights, learnRate):
    for t in range( len(actionList) ):
        action = actionList[t]

        for i in range( len(outputList[t]) ):
            correct = 1 - abs(action - i)
            dEdn = outputList[t][i] - correct

            for j in range( len(inputList[t]) ):
                dndw = inputList[t][j]*outputList[t][i]*(1 - outputList[t][i])
                dEdw = dEdn * dndw
                weights[i][j] -= learnRate * dEdw


def main():
    #creates polecart enviroment
    env = gym.make('CartPole-v0')

    #properties of the neural network
    inputNodes      = [0,0,0,0]
    outputNodes     = [0,0]
    weights         = nprnd.rand(2,4)*2 - 1
#    biases          = nprnd.random(2)*2 - 1
    learnRate       = 10

    #stores history of inputs and corresponding outputs and actions performed
    inputList  = []
    outputList = []
    actionList = []
    oppActionList = []

    #stores previous and current rewards (equivalent to time)
    avgReward  = 0
    reward     = 0
    rewardList = []

    for i_episode in range(200):
        #resets polecart enviroment
        observation = env.reset()

        for t in range(1000):
            #renders next timestep of polecart simulation
            env.render()

            #updates input and output nodes, then selects an action
            inputNodes  = observation
            outputNodes = setChoices(inputNodes, weights)
            action      = makeChoice(outputNodes)

            #updates values for current timestep
            observation, reward, done, info = env.step(action)

            #records state of neural net and action at the current timestep
            inputList.append(inputNodes)
            outputList.append(outputNodes)
            actionList.append(action)
            oppActionList.append( abs(action-1) )

            #reports length of simulation and starts a new simulation
            if done:
                print("Simulation finished after {} timesteps".format(t+1))
                reward = t+1
                rewardList.append(reward)
                break

        if reward >= avgReward:
            learn(inputList, outputList, actionList, weights, learnRate)
        elif reward < avgReward:
            learn(inputList, outputList, oppActionList, weights, learnRate)

        #resets inputList, outputList, actionList, and updates last reward
        inputList  = []
        outputList = []
        actionList = []
        oppActionList = []
        avgReward  = sum(rewardList)/len(rewardList)

    runList = [i for i in range( len(rewardList) )]
    z = np.polyfit(runList, rewardList, 1)
    p = np.poly1d(z)
    plt.plot(runList, p(runList), "r--")

    plt.scatter(runList, rewardList)
    plt.show()

if __name__ == "__main__":
    main()
