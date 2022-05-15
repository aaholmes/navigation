import pandas as pd
import matplotlib.pyplot as plt

for i in range(5):
    df1 = pd.read_csv('results/dqn/' + str(i) + '.txt')
    df1["avg"] = df1.rolling(100, min_periods=1).mean()
    plt.plot(df1["avg"], 'g')
    df1 = pd.read_csv('results/ddqn/' + str(i) + '.txt')
    df1["avg"] = df1.rolling(100, min_periods=1).mean()
    plt.plot(df1["avg"], 'r')
    df1 = pd.read_csv('results/ddqn_rand/' + str(i) + '.txt')
    df1["avg"] = df1.rolling(100, min_periods=1).mean()
    plt.plot(df1["avg"], 'b')
plt.title("Episodes to solve Bananas environment")
plt.xlabel("Episode number")
plt.ylabel("Average score (over past 100 episodes)")
plt.legend(['Vanilla DQN', 'Double DQN, Mean(Q1, Q2)', 'Double DQN, RandomChoice(Q1, Q2)'], loc='lower right')
plt.savefig('Results.png')
