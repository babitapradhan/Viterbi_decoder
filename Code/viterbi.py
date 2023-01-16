# implementation of viterbi algorithm

import sys
import pandas as pd
import copy
import random
import numpy as np


class Viterbi:

	def __init__(self, graph):
		# Represents the state machine graph
		self.__graph_dict = graph

	def encode(self, msg_bits):
		# Represent encoding through state machine
		state = 'S0'
		encoded_bits = ''
		for bit in msg_bits:
			# index 1 represents output bits
			encoded_bits = encoded_bits + self.__graph_dict[state][int(bit)][1]		
			state = self.__graph_dict[state][int(bit)][0] # index 0 represents state						
		return encoded_bits

	def __hamming_distance(self, code1, code2):
		# Sanity check for length of code1 and code2
		if(len(code1) != len(code2)):
			print ('Len(code1) != Len(code2)')
			sys.exit()

		dist = 0
		for bit1, bit2 in zip(code1, code2):
			if bit1 != bit2:
				dist = dist + 1
		return dist

	def __backward_connected_vertex(self, current_state):
		# Provides previous state information given the current state
		v = []
		# Iterate through all states, key represents prev_state
		for prev_state in self.__graph_dict:
			next_state_infos = self.__graph_dict[prev_state]
			for i in range(len(next_state_infos)):
				# check if state is present in prev_state
				if next_state_infos[i][0] == current_state:
					# next_state_infos[i][1] represents the output from prev_state to next_state
					v.append((prev_state, next_state_infos[i][1]))		
		return v

	def __next_state(self, prev_state):
		# Given prev_state, which are the possible next_states
		return [self.__graph_dict[prev_state][i][0] for i in range(len(self.__graph_dict[prev_state]))]

	def __msg_bit_extract(self, prev_state, current_state):
		# Given prev_state and current state, what is the transitioning message bit
		state_infos = self.__graph_dict[prev_state]
		for i in range(len(state_infos)):
			if state_infos[i][0] == current_state:
				return str(i)

		# Sanity Check
		print ('Bit not decoded, something is wrong. prev_state ', prev_state, ' and current_state ', current_state, 'is not possible')
		sys.exit()


	def decode(self, received_bits):
		# Decode received bits

		# Group received bits into group of 3 (since k/n = 1/3)
		received_bits = [received_bits[i:i+3] for i in range(0, len(received_bits), 3)]
		#print 'Received bits: ', received_bits
		#print 'No of symbols: ', len(received_bits)
		
		columns = ['S_' + str(i + 1) for i in range(len(received_bits))]
		# Cost table maintained in df_cost dataframe
		df_cost = pd.DataFrame(index = ['S0', 'S1', 'S2', 'S3'], columns = columns)
		# Trace src records best (min cost) prev_state given current state
		df_trace_src = pd.DataFrame(index = ['S0', 'S1', 'S2', 'S3'], columns = columns)

		# cost initialization
		cost = {'S0': 0, 'S1': 0, 'S2': 0, 'S3': 0}	
		trace_src = {'S0': '', 'S1': '', 'S2': '', 'S3': ''}	

		# First Stage (special case)
		state_first = 'S0'
		prev_state = state_first
		cost_new = copy.deepcopy(cost)
		trace_src_new = copy.deepcopy(trace_src)
		for i in range(len(self.__graph_dict[prev_state])):
			next_state = self.__graph_dict[prev_state][i]
			cost_new[next_state[0]] = cost[prev_state] + self.__hamming_distance(next_state[1], received_bits[0])
			trace_src_new[next_state[0]] = prev_state
		cost = cost_new
		# populate data and source location into dataframes
		df_cost['S_1'] = pd.Series(cost)
		trace_src = trace_src_new
		df_trace_src['S_1'] = pd.Series(trace_src)

		# Second Stage (special case)
		cost_new = copy.deepcopy(cost)
		trace_src_new = copy.deepcopy(trace_src)
		for prev_state in self.__next_state(state_first):
			#print 'Second State: ', prev_state
			for i in range(len(self.__graph_dict[prev_state])):
				next_state = self.__graph_dict[prev_state][i]
				cost_new[next_state[0]] = cost[prev_state] + self.__hamming_distance(next_state[1], received_bits[1])
				trace_src_new[next_state[0]] = prev_state
		cost = cost_new
		df_cost['S_2'] = pd.Series(cost)
		trace_src = trace_src_new
		df_trace_src['S_2'] = pd.Series(trace_src)

		# Rest Stages (general case)
		for stage in range(2, len(received_bits)):
			cost_new = copy.deepcopy(cost)
			trace_src_new = copy.deepcopy(trace_src)

			cost_new_new = {}
			trace_src_new_new = {}
			for this_state in self.__graph_dict:
				cost_temp = []
				
				for prev_state in self.__backward_connected_vertex(this_state):
					cost_temp.append(cost_new[prev_state[0]] + self.__hamming_distance(prev_state[1], received_bits[stage]))
				#print 'cost_temp: ', cost_temp
				cost_new_new[this_state] = min(cost_temp)
				trace_src_new_new[this_state] = self.__backward_connected_vertex(this_state)[cost_temp.index(min(cost_temp))][0]

			cost = cost_new_new
			trace_src = trace_src_new_new
			df_cost['S_' + str(stage + 1)] = pd.Series(cost)
			df_trace_src['S_' + str(stage + 1)] = pd.Series(trace_src)
		#print('cost table')
		#print (df_cost)
		#print('state table')
		#print (df_trace_src)

		# Trace-Back
		start_state = 'S0'
		current_state = start_state
		msg_bits = []
		for i in reversed(range(1, len(received_bits) + 1)):
			# .loc gives index of df and ['S_' + str(i)] gives column name of df
			prev_state = df_trace_src.loc[current_state]['S_' + str(i)]
			msg_bit = self.__msg_bit_extract(prev_state = prev_state, current_state = current_state)
			msg_bits.append(msg_bit)
			current_state = prev_state

		return ''.join(msg_bits[::-1])

	def error(self, original_msg_bits, decoded_msg_bits):
		return 1.0/len(original_msg_bits) * self.__hamming_distance(original_msg_bits, decoded_msg_bits)

	def add_noise(self, encoded_bits, pe):
		# BSC channel
		received_bits = ''
		flip_bits = "".join(np.random.choice(['1', '0'], len(encoded_bits), p = [pe, 1 - pe]))

		#for flip bits = 1, invert encoded bit
		for i in range(len(encoded_bits)):
			if flip_bits[i] == '1':
				if encoded_bits[i] == '1':
					received_bits = received_bits + '0'
				else:
					received_bits = received_bits + '1'
			else:
				received_bits = received_bits + encoded_bits[i]
		return received_bits, flip_bits
