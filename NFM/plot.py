import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data = pd.read_csv('NFM.csv')
# plt.title('NFM')
# plt.xlabel('step')
# plt.ylabel('loss')
#
# plt.plot(data.step,data.valid_auc,'b',label='valid_loss')
# plt.plot(data.step,data.train_auc,'r',label='train_loss')
# plt.legend(bbox_to_anchor=[0.3, 0.4])
# plt.yticks(np.linspace(0.45,0.7,20))
# plt.grid()
# plt.show()

data1 = pd.read_csv('NFM_all_embedding.csv')
data2 = pd.read_csv('DeepFM_all_embedding.csv')
data3 = pd.read_csv('FM_all_embedding.csv')
data4 = pd.read_csv('DNN_with_num.csv')

plt.title('Model loss - all embedding')
plt.xlabel('step')
plt.ylabel('loss')

plt.plot(data1.step[5:],data1.valid_auc[5:],'b',label='NFM loss')
plt.plot(data2.step[5:],data2.valid_auc[5:],'g',label='DeepFM loss')
# plt.plot(data3.step[5:],data3.valid_auc[5:],'r',label='FM loss')
# plt.plot(data4.step[5:],data4.valid_auc[5:],'y',label='DNN loss')
plt.legend(bbox_to_anchor=[0.3, 0.4])
plt.yticks(np.linspace(0.45,0.7,20))
plt.grid()
plt.show()