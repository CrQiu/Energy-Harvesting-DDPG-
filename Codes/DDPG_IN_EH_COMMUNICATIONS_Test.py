
# coding: utf-8

# In[1]:


"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
P2P, net bit rate, energy harvesting example for validation.
Using:
tensorflow 1.0
"""
import tensorflow as tf
import numpy as np
import gym
import time
import EH_P2P
import math

MAX_EPISODES = 1
MAX_EP_STEPS = 100000

RENDER = False
OUTPUT_GRAPH = True

env=EH_P2P.EH_P2P()
env.Chanpower()
env.Solarread()
B=np.zeros((100,4))    

def dense(x,a,b):
    results=(x.dot(a)+b)
    return results

def choose_action(s):
    a=dense(s,a1,b1)
    a=dense(a,a3,b3)
    a=dense(a,aa,ba)
    a=1/(1+np.exp(-a))
    a = np.clip(a, 0, 1)
    return a

for snr in range (-10,-8,2):

    for epoch in range (0,1750,30):

        modulation=0

        tf.reset_default_graph()
        graph = tf.get_default_graph()
        
        saver = tf.train.import_meta_graph("folder_for_nn_noise"+"/EH_save_net_snr="+str(snr)+str(modulation)+"epoch="+str(epoch)+"_P2P.ckpt.meta")
        with tf.Session() as sess:
            saver.restore(sess,"folder_for_nn_noise"+"/EH_save_net_snr="+str(snr)+str(modulation)+"epoch="+str(epoch)+"_P2P.ckpt")

            a1 = sess.run('Actor/eval_net/l1/kernel:0')
            b1 = sess.run('Actor/eval_net/l1/bias:0')
            a3 = sess.run('Actor/eval_net/l3/kernel:0')
            b3 = sess.run('Actor/eval_net/l3/bias:0')
            aa = sess.run('Actor/eval_net/a/a/kernel:0')
            ba = sess.run('Actor/eval_net/a/a/bias:0')
    

    
            for i in range(MAX_EPISODES):

                s = env.reset_P2P(snr=snr)
                s=np.reshape(s,(1,-1))
                ep_reward = 0
                for j in range(MAX_EP_STEPS):
                    s=np.array(s,dtype=float)
                    a=choose_action(s)
                    s_, r, info = env.step_P2P([a,modulation])
                    s = s_
                    ep_reward += r
                    if (j+1)%10000==0:
                        print("net bit rate=",ep_reward/j,"snr=",snr,"modulation=",modulation, "loop =",i,"action=",a,"noise=",env.noise)
          
                index=(snr+10)/2
                B[int(index),int(modulation)]=ep_reward/j
                print(B[int(index),int(modulation)])


# In[ ]:


print(B)


# In[ ]:


np.savetxt("DDPG_P2P_noise2"+str(snr)+".csv", B, delimiter = ',')

