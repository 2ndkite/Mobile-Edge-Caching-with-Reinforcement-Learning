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

#gym.hy1 version

class Agent(gym.Env, list):
    def __init__(self):
        print("Simple Mobile Edge Caching problem")
        self.observation_space = spaces.Box(low=-10, high=100, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(10) # update, replace&update
        self.action_space.n = 10
        np.random.seed(3)
        
        #reset content popularity equation Asin(Cx)+B
        a = np.random.randint(0, 50, size = 10)
        b = np.random.randint(0, 50, size = 10)
        for i in range(10):
            if (b[i]-a[i] < 0):
                b[i] = b[i] + np.random.randint(a[i]-b[i], a[i]-b[i]+5)
        c = np.random.rand(10)
        
        #content popularity reset making ndarray of request
        self.request = []
        for i in range(10):
            x = np.linspace(0,8*np.pi, 50)
            y = a[i]*np.sin(c[i]*x)+b[i]
            #plt.plot(x,y)
            #plt.show()
            int_y = list(map(int, y))
            self.request.append(int_y)
        self.request = np.array(self.request).T
        #print(self.request)
        
        
    def reset(self):
        # reset setup
        self.ts = 0
        self.done = False
        self.state = [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cache = [0, 0, 0, 0]
        self.reward = 0
        
        # state cache time step 0 setup
        self.cache = list(np.argsort(self.request[0])[::-1][0:4] + 1)

        for i in self.cache:
            self.state[i-1] = 1
        self.state[10:20] = self.request[0]/self.request[0].sum()
        #print(self.state)
        
        return self.state

    def step(self, action_value, eps = 0.0):
        #time step plus
        self.ts += 1
        #print(self.ts)
        #print("before")
        #print(self.state[10:20])
        #print(self.cache)
        # popularity update
        for i in range(10):
            if self.state[i] != -10:
                self.state[i] += 1  
        self.state[10:20] = self.request[self.ts]/self.request[self.ts].sum()
        
        #print("after popularity update")
        #print(self.state)
        #print(np.argsort(self.state[10:20])+1)
        #print("action_value")
        #print(action_value+1)
        
        #for the baseline result 
        self.state[self.cache[(self.ts-1)%4]-1] = 1
        #print(self.state)
        
        """
        # epsilon-greedy processed argmax index value = action_value
        if (action_value+1) in self.cache:   #index starts with zero
            # update AoI existing content
            self.state[action_value] = 1    
        else:
            dummy = []
            for i in range(4):
                dummy.append(self.state[self.cache[i]+9])
            #print("dummy sort")
            #print(np.argsort(dummy))
            #print("remove")
            #print(self.cache[np.argmin(dummy)])
            self.state[(self.cache[np.argmin(dummy)]-1)] = -10
            self.state[action_value] = 1
            new_cache = action_value + 1
            self.cache[np.argmin(dummy)] = new_cache
            #self.cache.remove(self.cache[np.argmin(dummy)])
            #self.cache.append(action_value + 1)
            self.state[action_value] = 1
        #print("after")
        #print(self.cache)
        #print(self.state)
        """
        
        # reward
        for i in range(10):
            if self.state[i] >= 0:
                self.reward += self.state[i+10]*(10 - self.state[i]-1)
            else:
                self.reward += self.state[i+10]*0
                
        #print(self.ts)
        #print(self.reward)
        
        if self.ts == 49:
            self.done = True
        return self.state, self.reward, self.done
            