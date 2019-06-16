
# coding: utf-8

# In[8]:

import math
import numpy as np
import csv
import random
from scipy.integrate import quad


def qfunc(x):
        y=0.5*math.erfc(np.sqrt(x/2))
        return y


class EH_P2P(object):
    def __init__(self):
        #######communication settings#######
        self.duration=5*60
        self.Xm=[2,3,4]
        self.alpha=np.array([[1,0],[2/3,2/3],[0.75,0.5]])
        aa=2*((math.sin(math.pi/8))**2)
        bb=2*((math.sin(3*math.pi/8))**2)
        self.beta=np.array([[1,1],[aa,bb],[0.2,1.8]])
        self.Ls=1000
        self.Tp=0.01
        self.Nor_DopplerFreq= 0.05
        self.Max_DopplerFreq= self.Nor_DopplerFreq/self.duration   # unit: Hz
        self.Observe_Data_Iter_Num=(72000)*10
        self.Discount_Factor= 0.99
        
        ############energy settings#########
        self.Solar_cell_size=4*0.2
        self.capacity=12*40*self.duration
        self.state = None
        self.i = 0     #epoch#
                
        ###########one way settings###########
        self.snr2=35
        self.noise2=40/10**(self.snr2/10)
        
          
        
    def Solarread(self):#2010-2011#

        with open("MDP_Optimal_Policy_BER - Multiple Power - Selective DF/ec201006.csv", "r") as csvfile:
            reader = csv.reader(csvfile) # 
            column = [row[5] for row in reader]
            column = np.array(column,dtype=float)

            column = column*0.01
            column = np.reshape(column,(30,-1)).T

            column = column[np.arange(12*7,12*17),:]

        with open("MDP_Optimal_Policy_BER - Multiple Power - Selective DF/ec201106.csv", "r") as csvfile:
            reader = csv.reader(csvfile) # 
            column2=[row[5] for row in reader]
            column2 = np.array(column2,dtype=float)

            column2=column2*0.01
            column2=np.reshape(column2,(30,-1)).T

            column2= column2[np.arange(12*7,12*17),:]


            column3 = np.hstack((column,column2))
            column3 = np.reshape(column3.T,(-1,1))
            column3 = np.tile(column3,(70,1))
            column3 = column3*10*self.duration*self.Solar_cell_size        
            column3= np.maximum(column3,10e-4*np.ones((column3.shape[0],column3.shape[1])))
            
            
            self.solar_sequence = column3
            self.SD_channel_sequence=self.channel_sequence[0:72000]
            self.SR_channel_sequence=self.channel_sequence[72000:72000*2]
            self.RD_channel_sequence=self.channel_sequence[72000*2:72000*3]
            self.channel_1_sequence=self.channel_sequence[0:72000]
            self.channel_2_sequence=self.channel_sequence[72000:72000*2]


        return column3
        csvfile.close()
        
    def Solartest(self):
        with open("MDP_Optimal_Policy_BER - Multiple Power - Selective DF/ec201206.csv", "r") as csvfile:
            reader = csv.reader(csvfile) # 
            column = [row[5] for row in reader]
            column = np.array(column,dtype=float)

            column = column*0.01
            column = np.reshape(column,(30,-1)).T

            column = column[np.arange(12*7,12*17),:]#7am to 5pm#
            column = np.reshape(column,(-1,1))
            column3 = np.tile(column,(41,1))               
            
            self.solar_sequence = column3*10*self.duration*self.Solar_cell_size
            column3= np.maximum(column3,10e-4*np.ones((column3.shape[0],column3.shape[1])))


            self.SD_channel_sequence=self.channel_sequence[0:72000]
            self.SR_channel_sequence=self.channel_sequence[72000:72000*2]
            self.RD_channel_sequence=self.channel_sequence[72000*2:72000*3]
            self.channel_1_sequence=self.channel_sequence[0:72000]
            self.channel_2_sequence=self.channel_sequence[72000:72000*2]

     

        return column3
        csvfile.close()


    def solarnoise(self):
        self.solar_sequence += 1*np.random.randn(self.solar_sequence.shape[0],self.solar_sequence.shape[1])          
        self.solar_sequence= np.maximum(self.solar_sequence,np.zeros((self.solar_sequence.shape[0],self.solar_sequence.shape[1])))
            



    def Chanpower(self):#Jakes model#
        n0= 100
        np2= (2*n0+1)*2
        wm= 2*math.pi*self.Max_DopplerFreq
        rp= 2.0*math.pi*np.random.rand(1,n0)
        Bn= math.pi*np.arange(1,n0+1)/n0
        Wn= wm*np.cos(2*math.pi*np.arange(1,n0+1)/np2)
        tt= np.arange(0,(self.Observe_Data_Iter_Num))*self.duration
        xc1_temp= np.kron(np.reshape(np.ones(np.size(tt)),(1,-1)),np.cos(np.reshape(Bn,(1,-1)).T))*np.cos(np.reshape(Wn,(1,-1)).T.dot(np.reshape(tt,(1,-1)))+np.kron(np.ones(np.size(tt)),rp.T)) 
        xs1_temp= np.kron(np.reshape(np.ones(np.size(tt)),(1,-1)),np.sin(np.reshape(Bn,(1,-1)).T))*np.cos(np.reshape(Wn,(1,-1)).T.dot(np.reshape(tt,(1,-1)))+np.kron(np.ones(np.size(tt)),rp.T)) 
        xc1= sum(xc1_temp)
        xs1= sum(xs1_temp)
        xc= 2.0/np.sqrt(np2)*np.sqrt(2.0)*xc1 + 2.0/np.sqrt(np2)*np.cos(math.pi/4)*np.cos(wm*tt)
        xs= 2.0/np.sqrt(np2)*np.sqrt(2.0)*xs1 + 2.0/np.sqrt(np2)*np.sin(math.pi/4)*np.cos(wm*tt)
        Observe_Channel_Sequence= xc**2+xs**2 # instantaneous channel power 
        #print(numpy.mean(Observe_Channel_Sequence),xc1.shape)
        self.channel_sequence = Observe_Channel_Sequence
        return Observe_Channel_Sequence
    
    

    
    def search_P2P(self,act): #for Lyapunov optimization#
        state=self.state
        solar,channel,battery=state       
        battery2=battery*100*self.duration
        channel2=(1+channel)
        action,Modulation_type=act
        error_rate=self.alpha[Modulation_type,0]*qfunc(self.beta[Modulation_type,0]*action*battery2*channel2/self.noise/self.duration)+self.alpha[Modulation_type,1]*qfunc(self.beta[Modulation_type,1]*action*battery2*channel2/self.noise/self.duration)
        reward=(self.Xm[Modulation_type]*self.Ls/self.Tp)*(1-error_rate)**(self.Xm[Modulation_type]*self.Ls)
        return  reward, {}



    def search_1_way2(self,act):  #for Lyapunov optimization#
        state=self.state
        action,Modulation_type=act
        solar,SD_channel,SR_channel,RD_channel,battery=state     

        battery2=battery*1000*self.duration

        SD_channel2=SD_channel+1
        SR_channel2=SR_channel+1
        RD_channel2=RD_channel+1

        Decoding_Capability_dB= 15; # 30;15;  (dB)
        Decoding_Capability= 10**(Decoding_Capability_dB/10)
        
        Source_Transmit_Power2= 40

        if (Source_Transmit_Power2<Decoding_Capability/SR_channel*self.noise2):
            SNR=Source_Transmit_Power2*SD_channel2/self.noise   
        else:
            SNR=(Source_Transmit_Power2*SD_channel2+action*battery2*RD_channel2/self.duration)/self.noise

        error_rate=self.alpha[Modulation_type,0]*qfunc(self.beta[Modulation_type,0]*SNR)+self.alpha[Modulation_type,1]*qfunc(self.beta[Modulation_type,1]*SNR)
            
        reward=(self.Xm[Modulation_type]*self.Ls/self.Tp)*(1-error_rate)**(self.Xm[Modulation_type]*self.Ls)
        return  reward, {}



    

    def step_P2P(self,act):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        ##recover from normalization##
        state=self.state
        solar,channel,battery=state       
        solar2=solar+0.1       
        solar2*=100*self.duration
        battery2=battery*100*self.duration
        channel2=channel+1
        ##recover from normalization##
        action,Modulation_type=act
        SNR=action*battery2*channel2/self.noise/self.duration

        error_rate=self.alpha[Modulation_type,0]*qfunc(self.beta[Modulation_type,0]*SNR)+self.alpha[Modulation_type,1]*qfunc(math.sqrt(self.beta[Modulation_type,1]*SNR))
            
        reward=(self.Xm[Modulation_type]*self.Ls/self.Tp)*(1-error_rate)**(self.Xm[Modulation_type]*self.Ls)#/100-200

        self.i+=1
        #normalize#
        battery=np.minimum(battery2*(1-action)+solar2,self.capacity)/self.duration/100
        channel=(self.channel_sequence[self.i])-1
        solar=self.solar_sequence[self.i]/self.duration/100-0.1
        self.state=solar,(channel),battery
        #normalize#
        return np.array(self.state), reward, {}    

    

    def judge_1_way(self):#whether the relay is on or off#
        state=self.state
        solar,SD_channel,SR_channel,RD_channel,battery=state


        solar2=solar*self.duration

        battery2=battery*self.duration

        SD_channel2=SD_channel
        SR_channel2=SR_channel
        RD_channel2=RD_channel
        Num_ModulationType= 4; # M-PSK: 4(Q)-PSK 
        G_MPSK_Modulation= math.sin(math.pi/Num_ModulationType)**2
        Decoding_Capability_dB= 15; # 30;15;  (dB)
        Decoding_Capability= 10**(Decoding_Capability_dB/10)
        
        Source_Transmit_Power= 40
        if (Source_Transmit_Power<Decoding_Capability/SR_channel*self.noise2):
            return 0 
        else:
            return 1



    def step_1_way2(self,act):
        state=self.state
        solar,SD_channel,SR_channel,RD_channel,battery=state
        flag=0
        action,Modulation_type=act
        solar2=solar+0.1
        solar2*=100*self.duration
        battery2=battery*100*self.duration
        SD_channel2=SD_channel+1
        SR_channel2=SR_channel+1
        RD_channel2=RD_channel+1


        Decoding_Capability_dB= 15; # 30;15;  (dB)
        Decoding_Capability= 10**(Decoding_Capability_dB/10)
        
        Source_Transmit_Power2= 40

        if (Source_Transmit_Power2<Decoding_Capability/SR_channel*self.noise2):
            SNR=Source_Transmit_Power2*SD_channel2/self.noise
       
        else:
            SNR=(Source_Transmit_Power2*SD_channel2+action*battery2*RD_channel2/self.duration)/self.noise

        error_rate=self.alpha[Modulation_type,0]*qfunc(self.beta[Modulation_type,0]*SNR)+self.alpha[Modulation_type,1]*qfunc(math.sqrt(self.beta[Modulation_type,1]*SNR))
            
        reward=(self.Xm[Modulation_type]*self.Ls/self.Tp)*(1-error_rate)**(self.Xm[Modulation_type]*self.Ls)
        self.i+=1

        battery=np.minimum(battery2*(1-action)+solar2,self.capacity)/self.duration/100
        SD_channel=self.SD_channel_sequence[self.i]-1
        SR_channel=self.SR_channel_sequence[self.i]-1
        RD_channel=self.RD_channel_sequence[self.i]-1
        solar=self.solar_sequence[self.i]/self.duration/100-0.1
        self.state=solar,SD_channel,SR_channel,RD_channel,battery
        return np.array(self.state), reward, flag, {}



    def reset_P2P(self,snr):
        self.i=0
        self.state = np.zeros((3,))
        self.noise=10**(-snr/10)
        self.state=self.solar_sequence[self.i+1]/100/self.duration-0.1,(self.channel_sequence[self.i]-1),self.solar_sequence[self.i]/100/self.duration
        return np.array(self.state)

     

    def reset_1_way(self,snr):
        self.state = np.zeros((5,))
        self.i=0
        self.state=self.solar_sequence[self.i+1]/100/self.duration-0.1,self.SD_channel_sequence[self.i]-1,self.SR_channel_sequence[self.i]-1,self.RD_channel_sequence[self.i]-1,self.solar_sequence[self.i]/100/self.duration
        self.noise=40/10**(snr/10)
        
        return np.array(self.state)

