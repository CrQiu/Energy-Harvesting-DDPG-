
# coding: utf-8

# In[ ]:


"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
1-way relay, net bit rate, energy harvesting example for validation.
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
MAX_EP_STEPS = 70000

RENDER = False
OUTPUT_GRAPH = False

np.random.seed(1)
tf.set_random_seed(1)

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

for modulation in range(1):
    for snr in range(0,22,2):

        tf.reset_default_graph()      
        graph = tf.get_default_graph()        
        saver = tf.train.import_meta_graph("folder_for_1way_net_bit_rate"+"/EH_save_net_snr="+str(0)+str(modulation)+"_1way.ckpt.meta")
       
        with tf.Session() as sess:
            
            saver.restore(sess,"folder_for_1way_net_bit_rate"+"/EH_save_net_snr="+str(0)+str(modulation)+"_1way.ckpt")
            a1 = sess.run('Actor/eval_net/l1/kernel:0')
            b1 = sess.run('Actor/eval_net/l1/bias:0')
            a3 = sess.run('Actor/eval_net/l3/kernel:0')
            b3 = sess.run('Actor/eval_net/l3/bias:0')
            aa = sess.run('Actor/eval_net/a/a/kernel:0')
            ba = sess.run('Actor/eval_net/a/a/bias:0')
   
            for i in range(MAX_EPISODES):

                ss=np.zeros((4,))
                ss_=np.zeros((4,))
                s = env.reset_1_way(snr)

                ss[0]=s[0]
                ss[1]=s[3]
                ss[2]=s[4]
                ss[3]=s[1]
                ep_reward = 0               
                
                for j in range(MAX_EP_STEPS):
 
                    ss=np.array(ss,dtype=float)
                    judge=env.judge_1_way()                
                    if judge == 1:                    
                        a=choose_action(ss) 
                        s_, r,_, info = env.step_1_way2([a,modulation])
                    else:
                        s_, r,_, info = env.step_1_way2([0,modulation])
                    ss_[0]=s_[0]
                    ss_[1]=s_[3]
                    ss_[2]=s_[4]
                    ss_[3]=s_[1]
                    ss = ss_
                    if r<0:
                        r*=-0.5
                    ep_reward += r                    
                    if (j+1)%10000==0:
                        print("net bit rate=",ep_reward/(j+1),"snr=",snr, "loop =",i,"action",a)
                index=(snr+10)/2
                B[int(index),int(modulation)]=ep_reward/j
                print(B[int(index),int(modulation)])


# In[ ]:


print(B)


# In[ ]:


np.savetxt("QPSK_1WAY_validation,snr=0"+str(snr)+".csv", B, delimiter = ',')

