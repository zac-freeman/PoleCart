import gym
import math
import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt

#bounds inputted x between (-1,1)
def sigmoid(x):
    return ( 1/(1 + np.exp(-x)) )

#selects a random output, favoring the largest one
def makeChoice(outputs):
    return int( outputs[1] > nprnd.random()*(outputs[1] + outputs[0]) )

#creates output nodes from input nodes and weights
def setChoices(inputs, weights, biases):
    outputs = [0,0]
    for i in range( len(outputs) ):
        outputs[i] += biases[i]

        for j in range( len(inputs) ):
            outputs[i] += weights[i][j]*inputs[j]
    return sigmoid(outputs[0]), sigmoid(outputs[1])

#updates weights and biases from outcomes of actions using linear regression
def learn(inputList, outputList, actionList, weights, biases, learnRate):
    for t in range( len(actionList) ):
        action = actionList[t]

        for i in range( len(outputList[t]) ):
            output = outputList[t][i]
            correct = 1 - abs(action - i)   #1 when weight or bias corresponds to correct output
            dEdn = output - correct
            dndb = output * (1 - output)
            dEdb = dEdn * dndb

            for j in range( len(inputList[t]) ):
                dndw = inputList[t][j] * output * (1 - output)
                dEdw = dEdn * dndw
                weights[i][j] -= learnRate * dEdw


def main():
    #creates polecart enviroment
    env = gym.make('CartPole-v0')

    #properties of the neural network
    inputNodes      = [0,0,0,0]
    outputNodes     = [0,0]
    weights         = nprnd.rand(2,4)*2 - 1
    biases          = nprnd.random(2)*2 - 1
    learnRate       = 10

    #stores history of inputs, outputs, actions, and opposite actions for one simulation
    #opposite actions are used in the case where a simulation is unsuccessful
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
            outputNodes = setChoices(inputNodes, weights, biases)
            action      = makeChoice(outputNodes)

            #updates values for current timestep
            observation, reward, done, info = env.step(action)

            #records state of neural net and action at the current timestep
            inputList.append(inputNodes)
            outputList.append(outputNodes)
            actionList.append(action)
            oppActionList.append( abs(action-1) )

            #reports length of simulation and ends simulation
            if done:
                print("Simulation finished after {} timesteps".format(t+1))
                reward = t+1
                rewardList.append(reward)
                break

        #encourages actions performed if simulation is successful
        #encourages opposite of actions performed if simulation is unsuccessful
        if reward >= avgReward:
            learn(inputList, outputList, actionList, weights, biases, learnRate)
        elif reward < avgReward:
            learn(inputList, outputList, oppActionList, weights, biases, learnRate)

        #resets inputList, outputList, actionList, and updates last reward
        inputList  = []
        outputList = []
        actionList = []
        oppActionList = []
        avgReward  = sum(rewardList)/len(rewardList)

    #plots a trendline of simulation lengths
    runList = [i for i in range( len(rewardList) )]
    z = np.polyfit(runList, rewardList, 1)
    p = np.poly1d(z)
    plt.plot(runList, p(runList), "r--")

    #plots a scatterplot of simulation lengths
    plt.scatter(runList, rewardList)
    plt.show()

if __name__ == "__main__":
    main()
