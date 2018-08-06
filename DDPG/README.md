# Deep Deterministic Policy Gradient

- **模型代码: DDPG.py**

DDPG是针对连续行为的策略学习方法,主要从：PG -> DPG -> DDPG 发展而来，具体算法不介绍。

**DDPG模型框架如下，actor-critic结构:**

<img src='1.jpeg'>





## DDPG对于DPG的关键改进

1. 使用卷积神经网络来模拟策略函数和Q函数，并用深度学习的方法来训练，证明了在RL方法中，非线性模拟函数的准确性和高性能、可收敛； 
   而DPG中，可以看成使用线性回归的机器学习方法：使用带参数的线性函数来模拟策略函数和Q函数，然后使用线性回归的方法进行训练。
2. experience replay memory的使用：actor同环境交互时，产生的transition数据序列是在时间上高度关联(correlated)的，如果这些数据序列直接用于训练，会导致神经网络的overfit，不易收敛。 
   DDPG的actor将transition数据先存入experience replay buffer, 然后在训练时，从experience replay buffer中随机采样mini-batch数据，这样采样得到的数据可以认为是无关联的。
3. target 网络和online 网络的使用， 使的学习过程更加稳定，收敛更有保障。

 

