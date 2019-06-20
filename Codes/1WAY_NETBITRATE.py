
# coding: utf-8

# In[1]:


"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
1-way relay, net bit rate, energy harvesting example for training.
Thanks to : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG
Using:
tensorflow 1.0
"""
import math
import tensorflow as tf
import numpy as np
import gym
import time
import EH_P2P
import DDPG_CLASS

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 1500
MAX_EP_STEPS = 240
LR_A = 0.0004  # learning rate for actor
LR_C = 0.0004  # learning rate for critic
GAMMA = 0.9   # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 40000
BATCH_SIZE = 80


OUTPUT_GRAPH = False

    
env=EH_P2P.EH_P2P()
env.Chanpower()
env.Solarread()
  

state_dim = 4 #SD_channel,RD_channel,battery，solar
action_dim = 1 #Transmission power
action_bound = 1 #no more than battery energy


if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

tip=1
tip2=1

for modulation in range(1):
    for snr in range(0,20,2):
        var = 10
        tip=1
        tip2=1
        tf.reset_default_graph()
        sess = tf.Session()
        with tf.name_scope('S'):
            S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
        with tf.name_scope('R'):
            R = tf.placeholder(tf.float32, [None, 1], name='r')
        with tf.name_scope('S_'):
            S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
        DDPG_CLASS.S=S
        DDPG_CLASS.R=R
        DDPG_CLASS.S_=S_
        actor= DDPG_CLASS.Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
        critic = DDPG_CLASS.Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
        actor.add_grad_to_graph(critic.a_grads)
        M = DDPG_CLASS.Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=100)        
        

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
 

                judge=env.judge_1_way()
                if judge == 1:
                    a = actor.choose_action(ss)
                    a = np.random.normal(a, var)
                    a=np.clip(a,0,1)

                    s_, r, flag,info = env.step_1_way2([a,modulation])#input modulation 0:qpsk,1:8psk,2:16qam
                    
                    ss_[0]=s_[0]
                    ss_[1]=s_[3]
                    ss_[2]=s_[4]
                    ss_[3]=s_[1]

                    M.store_transition(ss, a, (r), ss_)

                    if M.pointer > MEMORY_CAPACITY:
                        if tip == 1:
                            print("memory full",j,i)
                            tip=0
                        var *= 0.9995  # decay the action randomness
                        if tip2 == 1 and var<0.00000001:
                            print("var zero",j,i)
                            tip2=0
                        b_M = M.sample(BATCH_SIZE)
                        b_s = b_M[:, :state_dim]
                        b_a = b_M[:, state_dim: state_dim + action_dim]
                        b_r = b_M[:, -state_dim - 1: -state_dim]
                        b_s_ = b_M[:, -state_dim:]

                        critic.learn(b_s, b_a, b_r, b_s_)
                        actor.learn(b_s)

                else:     
                    a=-1
                    s_, r,flag , info = env.step_1_way2([0,modulation])#input modulation 0:qpsk,1:8psk,2:16qam  
                
                s = s_ 
                ss[0]=s[0]
                ss[1]=s[3]
                ss[2]=s[4]
                ss[3]=s[1]              

                ep_reward += r



            if i % 30 == 0 :
                print("Net bit rate=",r,"action",a, "solar,channel,battery",s,"average_reward",ep_reward/j)
              


        save_path = saver.save(sess, "folder_for_1way_net_bit_rate"+"/EH_save_net_snr="+str(snr)+str(modulation)+"_1way.ckpt")

        print("Save to path: ", save_path)
print("----------------------------END--------------------------------")

