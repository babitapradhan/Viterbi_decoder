import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from viterbi import Viterbi
import random

# State Diagram Graph
g = {'S0': [('S0', '000'), ('S1', '111')],
    'S1': [('S2', '101'), ('S3', '010')],
    'S2': [('S0', '011'), ('S1', '100')],
    'S3': [('S2', '110'), ('S3', '001')],
    }

# Program Parameters
K = 1024		# msg bit length
Iter = 100		# Max Iterations
Pb_list = np.arange(0, 0.5, 0.01) # BSC Channel Error

V = Viterbi(g)

Pe_list = []

for Pb in Pb_list:
    print ('Pb = ', Pb)
    decoded_error_list = []

    for iter in range(Iter):
        print ('\tIterations: ', iter)

        #randomly generate K message bits
        msg_bits = ''.join([random.choice('10') for _ in range(K)])

        #add flush bits to message bits
        msg_bits = msg_bits + "000"
        #print 'Msg bits:      ', msg_bits

        encoded_bits = V.encode(msg_bits)
        #print 'Encoded bits:  ', encoded_bits

        received_bits, flip_bits = V.add_noise(encoded_bits, Pb)
        #print 'Flip locations:', flip_bits
        #print 'Received bits: ', received_bits

        decoded_msg_bits = V.decode(received_bits)
        #print 'Decoded Bits:  ', decoded_msg_bits

        # Channel error calculation
        channel_error = V.error(encoded_bits, received_bits)
        print ('\tChannel Error Probability = ', channel_error)

        # decoding error calculation removing 3 flush bits
        decode_error = V.error(msg_bits[:-3], decoded_msg_bits[:-3])	
        decoded_error_list.append(decode_error) 

    #average error calculation
    mean_error = np.mean(decoded_error_list)
    print ('Average Error for BSC Pb = ', Pb, ': ', mean_error)
    Pe_list.append(mean_error)

print ('Decoded error: ', Pe_list)

df = pd.DataFrame.from_dict({'Pb': Pb_list, 'Pe': Pe_list})
ax = df.plot.line(x = 'Pb', y = 'Pe')
ax.set_ylabel('Pe')
ax.set_yscale('log')
plt.show()
