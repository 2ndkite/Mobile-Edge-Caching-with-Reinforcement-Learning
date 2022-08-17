import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import copy
import time
import logging
import math
import sys
import os
import numpy as np
from pdb import set_trace as debug
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


#gym.hy version

class Agent(gym.Env, list):
    def __init__(self):
        print("Simple Mobile Edge Caching problem")
        self.observation_space = spaces.Box(low=-50, high=100, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(8) # update, replace&update
        self.action_space.n = 8
        np.random.seed(1)
        
        #reset content popularity equation Asin(Cx)+B
        """
        a = np.random.randint(1, 10, size = 8)
        b = np.random.randint(1, 10, size = 8)
        for i in range(8):
            if (b[i]-a[i] < 0):
                b[i] = b[i] + np.random.randint(a[i]-b[i], a[i]-b[i]+5)
        """
        #c = np.random.rand(8)
        
        
        
        
        #content popularity generator
        c = np.random.randint(1, 5, size = 8)
        d = np.random.randint(1, 20, size = 8)
        self.request = []
        for i in range(8):
            x = np.linspace(0,8*np.pi, 50)
            y = 5*np.sin(c[i]*x+d[i])+5
            int_y = list(map(int, y))
            self.request.append(int_y)
        self.request = np.array(self.request).T
        
        
        print(self.request)
        for i in range(50):
            print(np.argmax(self.request[i])+1)
        
        
    def reset(self):
        # reset setup
        self.ts = 0
        self.done = False
        #self.state = [100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state = [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cache = [0, 0, 0]
        self.reward = 0
        
        # state cache time step 0 setup
        self.cache = list(np.argsort(self.request[0])[::-1][0:3] + 1)
        #print(self.cache)
        for i in self.cache:
            self.state[i-1] = 1
        self.state[8:16] = (self.request[0]/self.request[0].sum())*10
        #print(self.state)
        
        return self.state

    def step(self, action_value, eps = 0.0):

        #time step plus
        self.ts += 1
               
        # popularity update
        for i in range(8):
            if self.state[i] != -10:
                self.state[i] += 1  
        self.state[8:16] = (self.request[self.ts]/self.request[self.ts].sum())*10

            
        # epsilon-greedy processed argmax index value = action_value
        if (action_value+1) in self.cache:   #index starts with zero
            # update AoI existing content
            self.state[action_value] = 1      
        else:
            dummy = []
            for i in range(3):
                dummy.append(self.state[self.cache[i]+7])
            #self.state[(self.cache[np.argmin(dummy)]-1)] = -10
            self.state[(self.cache[np.argmin(dummy)]-1)] = -10
            self.state[action_value] = 1
            new_cache = action_value + 1
            self.cache[np.argmin(dummy)] = new_cache
        
                   
        """
        # update window
        for i in self.cache:
            if self.state[i-1] >= 5:
                self.state[i-1] = 1
        """
              
        
        #print(self.ts)
        #print(self.cache)
        #print(self.state[0:8])
        
        
        avg_aoi = 0
        # reward
        for i in range(8):
            if self.state[i] == -10:
                avg_aoi += self.state[i+8]/10 * 11
            else:
                avg_aoi += self.state[i+8]/10 * (self.state[i]+1) 
            if self.state[i] != -10:
                if self.state[i] < 10:
                    self.reward += self.state[i+8]/10 * (10 - self.state[i])
                else:
                    self.reward += self.state[i+8]/10 * (10 - self.state[i])
                
                
                
                
            """    
            if self.state[i] != -50:
                if self.state[i] + 50 < 20:
                    self.reward += self.state[i+8]/100*(self.state[i]+ 50 + 1)
                else:
                    self.reward += self.state[i+8]/100*(self.state[i])
            else:
                self.reward += self.state[i+8]/100 * (20 + 1)
            """    
                
                
            """    
            if self.state[i] > 19 and self.state[i] <= 50:
                self.reward -= self.state[i+8]/50*(20 - self.state[i] - 1)
            elif self.state[i] > 0 and self.state[i] <= 19 :
                self.reward += self.state[i+8]/50*(20 - self.state[i] - 1)
            elif self.state[i] == 100:
                self.reward += self.state[i+8]/50*(-1)
            """
            

        #print(self.ts)
        #print(self.reward)
        if self.ts == 49:
            self.done = True
        return self.state, self.reward, self.done, avg_aoi