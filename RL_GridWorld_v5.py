#!/usr/bin/env python 

import sys, time, random, csv
import numpy as np
from os import path
from datetime import datetime
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf,suppress=True,threshold=sys.maxsize,precision=5)

def readInput(fileName):
	if path.exists(fileName) == False: help("Incorrect file name")
	world = []
	f = open(fileName, 'r')
	lines = f.readlines()
	for line in lines:
		line = line.rstrip()
		world.append([float(item) for item in line.split(',')])
	return(np.asarray(world))

def help(text):
	print(text)
	print('\n-----RL Q-Learning Problem Help-----\n')
	print('Arguments : [GridFile] [MoveCost] [Probability] [TimeLimit]')
	print('GridFile\t: .csv file name with the grid world')
	print('MoveCost\t: Move cost')
	print('Probability\t: Probability that a move will go in the desired direction')
	print('TimeLimit\t: Timelimit to learn')
	print('\n-----End Help-----\n')
	sys.exit()

class Q_learn:
	def __init__(self,args,e,eDec):
		self.world = readInput(args[1])							# Grid World
		self.r = len(self.world)										# Number of rows in the grid world
		self.c = len(self.world[0])									# Number of columns in the grid world
		self.mCost = float(args[2])     						# Move Cost
		self.mProb = float(args[3])     						# Probability
		self.timeLimit = float(args[4])  						# Timelimit
		self.startState = np.array([self.r-1,0]) 		# Start at bottom left
		self.alpha = 0.8														# Step Size
		self.alphaMin = 0.05
		self.gamma = 1.0														# Discount Rate
		self.eps = e															  # Exploration probability
		self.epsDecay = eDec  									    # Exploration probability decay rate
		self.alphaDecay = 0.99999  									# Alpha decay rate
		self.Q_SA = np.zeros([self.r,self.c,4])			# Q(state,action)
		self.stopOnConverge = int(args[5])
		
		self.nIter = 0															# Number of learning iterations
		self.ctrStates = np.zeros([self.r,self.c])	# Tacking number of times each state was reached (Only for tracking)
		self.QRecord = []
		self.QRecordAvg = []
		self.Reward = []
		self.RewardAvg = []
		self.movingAvgParam = 2500
		
		# Actions : 0->UP, 1->RIGHT. 2->DOWN, 3->LEFT
		# Lookup table for policy
		self.lookup = ["\u2191","\u2192","\u2193","\u2190"]							
		self.writeFlag = 0													# Flag for writing output to text file results.txt

	def printConfig(self,text):
		print("-----",text)
		print("Gridworld :"); print(self.world)
		print("Move Cost \t\t:",self.mCost)
		print("Move Probability \t:",self.mProb)
		print("Start State \t\t:",self.startState)
		print("Step Size \t\t:",self.alpha)
		print("Step Size Decay Rate \t:",self.alphaDecay)
		print("Discount Rate \t\t:",self.gamma)
		print("Epsilon \t\t:",self.eps)
		print("Epsilon Decay Rate \t:",self.epsDecay)
		print("-----")
		if self.writeFlag==1:
			file = open("record.txt","a")
			now = datetime.now()
			file.write(now.strftime("%d/%m/%Y, %H:%M:%S")+"\n")
			file.write("-----"+text+"\n")
			file.write("Gridworld :"+"\n"); file.write(np.array_str(self.world)+"\n")
			file.write("Move Cost ="+str(self.mCost)+"\n")
			file.write("Move Probability ="+str(self.mProb)+"\n")
			file.write("Start State :"+str(self.startState)+"\n")
			file.write("Step Size :"+str(self.alpha)+"\n")
			file.write("Discount Rate :"+str(self.gamma)+"\n")
			file.write("Epsilon :"+str(self.eps)+"\n")
			file.write("Epsilon Decay Rate :"+str(self.epsDecay)+"\n")
			file.write("-----"+"\n")
			file.close()

	def printResult(self):
		policy = self.world.astype(str)
		maxQ 	 = self.world.astype(float)
		for i in range(self.r):
			for j in range(self.c):
				if self.world[i,j] == 0:
					policy[i,j] = self.lookup[np.argmax(self.Q_SA[i,j,:])]
					maxQ[i,j]		= np.amax(self.Q_SA[i,j,:])
				else:
					policy[i,j] = "\u25A9"

		# Color coding the policy for ease of visulaization
		red = "\x1b[31m";green = "\x1b[32m";off = "\x1b[0m"
		colorCode = self.world.astype(str)
		colorCode[:,:] = off
		colorCode[np.where(self.world>0)] = green
		colorCode[np.where(self.world<0)] = red

		print("Policy after",self.nIter,"iterations :");
		for i in range(self.r):
			for j in range(self.c):
					print(colorCode[i,j]+policy[i,j]+off, end =" ")
			print('\n', end ="")

		print("Max Q at start state :",maxQ[self.startState[0],self.startState[1]])
		# print("Max Q at each state :");print(maxQ)
		# print("Number of times each state was visited :");print(self.ctrStates)
		print("Final epsilon value :",round(self.eps,4))
		print("Final step size :",round(self.alpha,4))
		print("-----")
		if self.writeFlag==1:
			file = open("record.txt","a") 
			file.write("Policy after "+str(self.nIter)+" iterations :"+"\n");file.write(np.array_str(policy)+"\n")
			file.write("Max Q at start state :"+str(maxQ[self.startState[0],self.startState[1]])+"\n")
			file.write("Max Q at each state :"+"\n");file.write(np.array_str(maxQ)+"\n")
			file.write("Number of times each state was visited :"+"\n");file.write(np.array_str(self.ctrStates)+"\n")
			file.write("Final epsilon value :"+str(round(self.eps,4))+"\n")
			file.write("-----\n")
			file.close() 
 
	def transitionModel(self,state,action):
		# Action is probabilistic with probability = prob.
		# The remaining probability is split among moving 90 degrees left or right of the desired direction
		action += np.random.choice([-1,0,1],1,\
							p=[(1-self.mProb)/2,self.mProb,(1-self.mProb)/2])
		action %= 4		# Wrapping up so that action is between [0,3]
		
		if action == 0 and state[0] > 0: 						return(np.array([state[0]-1,state[1]]))
		elif action == 1 and state[1] < self.c-1: 	return(np.array([state[0],state[1]+1]))
		elif action == 2 and state[0] < self.r-1: 	return(np.array([state[0]+1,state[1]]))
		elif action == 3 and state[1] > 0:   				return(np.array([state[0],state[1]-1]))
		else:																				return(state)	

	def learner(self):
		sTime = time.time()
		runTime = 0
		while True:
			self.nIter += 1
			state = self.startState										# Setting state to the starting point
			self.ctrStates[state[0],state[1]] += 1		# Incrementing the tracking counter
			RewardIter = 0
			while(self.world[state[0],state[1]] == 0 and runTime <= self.timeLimit):
				# Choose between exploration and exploitation
				explore = np.random.choice([0,1],1,p=[1-self.eps,self.eps])

				# Select action based on the mode
				if explore == 1:	# Exploration
					action = np.random.choice([0,1,2,3],1)
				else:							# Exploitation
					action = np.argmax(self.Q_SA[state[0],state[1],:])

				nextState = self.transitionModel(state,action)
				Q 				= self.Q_SA[state[0],state[1],action]
				reward 		= self.world[nextState[0],nextState[1]] + self.mCost
				RewardIter += reward
				maxQNext	= np.amax(self.Q_SA[nextState[0],nextState[1],:])

				# Updating Q table
				self.Q_SA[state[0],state[1],action] = Q + self.alpha*(reward + self.gamma*maxQNext - Q)
				state = nextState

				self.ctrStates[state[0],state[1]] += 1		# Incrementing the tracking counter
				
				# Updating the exploration probability
				self.eps *= self.epsDecay
				if self.alpha>self.alphaMin:
					self.alpha *= self.alphaDecay

				runTime = time.time() - sTime

			self.QRecord.append(np.sum(np.abs(self.Q_SA)))
			self.QRecordAvg.append(np.average(self.QRecord[max(-len(self.QRecord),-self.movingAvgParam):]))
			self.Reward.append(RewardIter)
			self.RewardAvg.append(np.average(self.Reward[max(-len(self.Reward),-500):]))
			
			if self.stopOnConverge == 1 and self.nIter>self.movingAvgParam*2:
				converge = abs(self.QRecordAvg[-1] - self.QRecordAvg[-1-int(self.movingAvgParam/4)]) + \
									 abs(self.QRecordAvg[-1] - self.QRecordAvg[-1-int(self.movingAvgParam/2)])

				if self.eps < 0.50 and converge <= 0.025:
					print("Model has converged. Exiting...")
					print("Time Used :",round(runTime,4),"secs")
					break

			if runTime > self.timeLimit:
				if self.stopOnConverge == 1:
					print("Time limit reached. Model may not have converged. Exiting...")
					print("Time Used :",round(self.timeLimit,4),"secs")
				break

# --------Code starts here--------
if len(sys.argv) != 6: help("ERROR : Incorrect number of arguments")
QL1 = Q_learn(sys.argv,1.0,0.999995)
QL1.printConfig("Initial Configuration")
QL1.learner()
QL1.printResult()