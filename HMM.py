#======================================================================================
#
#
#						Implementation of HMM Model
#						Author: Neha Bhoi and Liam Nguyen 
#
#
#======================================================================================

import numpy as np
import pandas as pd


#load data from the file
def LoadData():
	dataset=pd.read_csv("./Project2Data.txt",delimiter=",",header=None)
	return dataset

# This fucntion creates a_ij matrix	
def CreateTransitionProbs(dataset):
	previous_row = []
	sunny_sunny , sunny_rainy , sunny_foggy = 0,0,0
	rainy_rainy , rainy_sunny , rainy_foggy = 0,0,0
	foggy_foggy , foggy_sunny , foggy_rainy = 0,0,0
	for index, row in dataset.iterrows():
		if(len(previous_row) != 0):
			if(previous_row[0] == "sunny" and row[0] == "sunny"):
				sunny_sunny = sunny_sunny + 1
			elif(previous_row[0] == "sunny" and row[0] == "foggy"):
				sunny_foggy = sunny_foggy + 1
			elif(previous_row[0] == "sunny" and row[0] == "rainy"):
				sunny_rainy = sunny_rainy + 1
			elif(previous_row[0] == "rainy" and row[0] == "rainy"):
				rainy_rainy = rainy_rainy + 1
			elif(previous_row[0] == "rainy" and row[0] == "foggy"):
				rainy_foggy = rainy_foggy + 1
			elif(previous_row[0] == "rainy" and row[0] == "sunny"):
				rainy_sunny = rainy_sunny + 1
			elif(previous_row[0] == "foggy" and row[0] == "foggy"):
				foggy_foggy = foggy_foggy + 1
			elif(previous_row[0] == "foggy" and row[0] == "rainy"):
				foggy_rainy = foggy_rainy + 1
			elif(previous_row[0] == "foggy" and row[0] == "sunny"):
				foggy_sunny = foggy_sunny + 1
			previous_row = row
		else:
			previous_row = row

	#Calculate individual transmisiion Probabiliteies 
	prob_sunny_sunny = sunny_sunny / (sunny_sunny + sunny_rainy + sunny_foggy)
	prob_sunny_rainy = sunny_rainy / (sunny_sunny + sunny_rainy + sunny_foggy)
	prob_sunny_foggy = sunny_foggy / (sunny_sunny + sunny_rainy + sunny_foggy)

	prob_rainy_rainy = rainy_rainy / (rainy_rainy + rainy_sunny + rainy_foggy)
	prob_rainy_sunny = rainy_sunny / (rainy_rainy + rainy_sunny + rainy_foggy)
	prob_rainy_foggy = rainy_foggy / (rainy_rainy + rainy_sunny + rainy_foggy)

	prob_foggy_foggy = foggy_foggy / (foggy_foggy + foggy_sunny + foggy_rainy)
	prob_foggy_sunny = foggy_sunny / (foggy_foggy + foggy_sunny + foggy_rainy)
	prob_foggy_rainy = foggy_rainy / (foggy_foggy + foggy_sunny + foggy_rainy)
    
	hidden_states = ['sunny','rainy','foggy']
	aij_df=pd.DataFrame(columns = hidden_states, index = hidden_states)
	aij_df.loc[hidden_states[0]] = [prob_sunny_sunny,prob_sunny_rainy,prob_sunny_foggy]
	aij_df.loc[hidden_states[1]] = [prob_rainy_sunny,prob_rainy_rainy,prob_rainy_foggy]
	aij_df.loc[hidden_states[2]] = [prob_foggy_sunny,prob_foggy_rainy,prob_foggy_foggy]
	return aij_df

# This fucntion creates b_jk matrix	
def CreateEmitionProbs(dataset):
	sunny_yes, sunny_no = 0 ,0
	rainy_yes, rainy_no = 0, 0
	foggy_yes, foggy_no = 0, 0

	for index, row in dataset.iterrows():
		if(row[0] == "sunny" and row[1] == "yes"):
			sunny_yes = sunny_yes + 1
		elif(row[0] == "sunny" and row[1] == "no"):
			sunny_no = sunny_no + 1
		elif(row[0] == "rainy" and row[1] == "yes"):
			rainy_yes = rainy_yes + 1
		elif(row[0] == "rainy" and row[1] == "no"):
			rainy_no = rainy_no + 1
		elif(row[0] == "foggy" and row[1] == "yes"):
			foggy_yes = foggy_yes + 1
		elif(row[0] == "foggy" and row[1] == "no"):
			foggy_no = foggy_no + 1

	#Calculate individual emission probabilities
	prob_sunny_yes = sunny_yes / (sunny_yes + sunny_no)
	prob_sunny_no = sunny_no / (sunny_yes + sunny_no)

	prob_rainy_yes = rainy_yes / (rainy_yes + rainy_no)
	prob_rainy_no = rainy_no / (rainy_yes + rainy_no)

	prob_foggy_yes = foggy_yes / (foggy_yes + foggy_no)
	prob_foggy_no = foggy_no / (foggy_yes + foggy_no)

	observable_states = ['yes','no']
	hidden_states = ['sunny','rainy','foggy']
        
	bjk_df = pd.DataFrame(columns=observable_states , index=hidden_states )
	bjk_df.loc[hidden_states[0]] = [prob_sunny_yes , prob_sunny_no]
	bjk_df.loc[hidden_states[1]] = [prob_rainy_yes , prob_rainy_no]
	bjk_df.loc[hidden_states[2]] = [prob_foggy_yes , prob_foggy_no]
	return bjk_df

#function convert visible state to numeric Values 
def convert_to_stateindex(visibleStateList):
    resultVisibleStateList =[]
    for x in visibleStateList:
        if x.lower() == "yes":
            resultVisibleStateList.append(0)
        else:
            resultVisibleStateList.append(1)
    return resultVisibleStateList

#function to calculate alpha using froward algorithm
def CreateAlphas(visibleStateList,aij_df,bjk_df):
	a = np.array(aij_df.values)
	b = np.array(bjk_df.values)
	initial_distribution=np.array((1,0,0))
	v = visibleStateList
	alpha = np.zeros((len(v), a.shape[0]))
	alpha[0, :] = initial_distribution * b[:, v[0]]
	val = ["T0"]
	for t in range(1, len(v)):
		val.append("T"+str(t))
		for j in range(a.shape[0]):
			alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, v[t]]
	
	alpha_df = pd.DataFrame(data=alpha, index = val, columns = ['W0:Sunny','W1:Rainy','W2:Foggy'])
	alpha_df = alpha_df.swapaxes("index", "columns") 
	print("=================== Alpha Matrix ====================\n",alpha_df,"\n")
	return alpha
		
#function to calculate hiden states using vertibri algorithm		
def RunVertibri(visibleStateList,aij_df,bjk_df):
	visbleStateListIndex = convert_to_stateindex(visibleStateList)
	forward_output = CreateAlphas(visbleStateListIndex,aij_df,bjk_df)
	
	hidden_state_output = []
	for i in range(len(forward_output)): 
		max_index = np.argmax(forward_output[i])
		hidden_state_output.append(hidden_states[max_index])
	return hidden_state_output
	
#main function
if __name__ == "__main__":
	dataset = LoadData()
	hidden_states = ['sunny','rainy','foggy']
	Aij_Matrix = CreateTransitionProbs(dataset)
	print("=========================== A_ij (Tarnsitions Probabilities) Matrix ==========================\n",Aij_Matrix,"\n")
	Bjk_Matrix = CreateEmitionProbs(dataset)
	print("=========================== B_jk (Emission Probabilities) Matrix ==========================\n",Bjk_Matrix,"\n")
	VT =  ['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
	output_state = RunVertibri(VT,Aij_Matrix,Bjk_Matrix)
	print("Input: Visibale State Sequence to HMM Model: ",VT)
	print("Output: Hidden State Sequence: ",output_state)




