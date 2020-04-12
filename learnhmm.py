import sys
import numpy as np
import scipy.special as sp

class HiddenMarkovModelLearner:
  """Code author: Ting-Sheng Lin (tingshel@andrew.cmu.edu)"""

  def __generateStateIndex(self, file_name):
    f_in = open(file_name, 'r')
    state_index = {}
    i = 0
    while True:
      state = f_in.readline().replace('\n','')
      if not state:
        break
      state_index[state] = i
      i += 1

    return state_index

  def __generateTrainData(self, file_name):
    f_in = open(file_name, 'r')
    obs_states_list, hid_states_list = [], []
    while True:
      row = f_in.readline()
      if not row:
        break

      tmp_obs_states, tmp_hid_states = [], []
      for word in row.split():
        obs, hid = word.split('_')
        tmp_obs_states.append(obs)
        tmp_hid_states.append(hid)
      obs_states_list.append(tmp_obs_states)
      hid_states_list.append(tmp_hid_states)

    return [obs_states_list, hid_states_list]

  def learn(self, train_file, obs_index_file, hid_index_file):
    obs_states_list, hid_states_list = self.__generateTrainData(train_file)
    obs_index = self.__generateStateIndex(obs_index_file)
    hid_index = self.__generateStateIndex(hid_index_file)
    K = len(hid_index) # num of category of hidden state y
    O = len(obs_index) # num of category of observed state x

    # initialize and do the pseudo first
    pi = np.full((K, 1), 1.0) # p(y_1), make it 2d array for output convenience
    pi_sum = K
    a = np.full((K, K), 1.0) # p(y_t | y_t-1)
    a_sum = np.full(K, K) # sum for each y 
    b = np.full((K, O), 1.0) # p(x_t | y_t)
    b_sum = np.full(K, O)

    for i in range(len(hid_states_list)):
      for t in range(len(hid_states_list[i])):
        if t == 0:
          pi[hid_index[hid_states_list[i][t]]][0] += 1
          pi_sum += 1
        else:
          a[hid_index[hid_states_list[i][t - 1]]][hid_index[hid_states_list[i][t]]] += 1
          a_sum[hid_index[hid_states_list[i][t - 1]]] += 1
        b[hid_index[hid_states_list[i][t]]][obs_index[obs_states_list[i][t]]] += 1
        b_sum[hid_index[hid_states_list[i][t]]] += 1

    for state in range(len(pi)):
      pi[state][0] = pi[state][0] / pi_sum
    
    for pre_state in range(len(a)):
      for next_state in range(len(a[pre_state])):
        a[pre_state][next_state] /= a_sum[pre_state]

    for pre_state in range(len(b)):
      for obs_state in range(len(b[pre_state])):
        b[pre_state][obs_state] /= b_sum[pre_state]
    
    return [pi, a, b]

  def __generateOutputStr(self, matrix):
    output_str = ""
    for probabilities in matrix:
      output_str += f"{' '.join([format(p, '.18e') for p in probabilities])}\n"
    return output_str

  def outputToFiles(self, matrix, file_name):
    output_str = self.__generateOutputStr(matrix)
    f_out = open(file_name,"w+")
    f_out.write(output_str)


if __name__ == '__main__':
  train_input = sys.argv[1]
  index_to_word = sys.argv[2]
  index_to_tag = sys.argv[3]
  hmmprior = sys.argv[4]
  hmmemit = sys.argv[5]
  hmmtrans = sys.argv[6]
  
  hmm = HiddenMarkovModelLearner()
  pi, a, b = hmm.learn(train_input, index_to_word, index_to_tag)
  hmm.outputToFiles(pi, hmmprior)
  hmm.outputToFiles(b, hmmemit)
  hmm.outputToFiles(a, hmmtrans)
