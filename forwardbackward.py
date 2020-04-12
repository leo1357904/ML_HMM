import sys
import numpy as np
import scipy.special as sp

class HiddenMarkovModelTester:
  """Code author: Ting-Sheng Lin (tingshel@andrew.cmu.edu)"""

  def __generateMatrix(self, file_name):
    f_in = open(file_name, 'r')
    matrix = []
    while True:
      row = f_in.readline()
      if not row:
        break
      tmp_probability_list = []
      for probability in row.split():
        tmp_probability_list.append(float(probability))
      matrix.append(tmp_probability_list)

    return matrix


  def __generateStateIndex(self, file_name):
    f_in = open(file_name, 'r')
    state_index = {}
    index_state = {}
    i = 0
    while True:
      state = f_in.readline().replace('\n','')
      if not state:
        break
      state_index[state] = i
      index_state[i] = state
      i += 1

    return [state_index, index_state]


  def __generateTestData(self, file_name):
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


  def test(self, test_file, obs_index_file, hid_index_file, prior_file, emit_file, trans_file, predict_file, metric_file):
    obs_index, index_obs = self.__generateStateIndex(obs_index_file)
    hid_index, index_hid = self.__generateStateIndex(hid_index_file)
    pi = self.__generateMatrix(prior_file)
    b = self.__generateMatrix(emit_file)
    a = self.__generateMatrix(trans_file)
    obs_states_list, hid_states_list = self.__generateTestData(test_file)
    N = len(obs_states_list) # num of data row
    K = len(hid_index) # num of hidden state y
    
    predict_str = ""
    predictions = []
    log_likelihood = 0
    for i in range(N): # each data row
      # forward
      alpha = []
      T = len(obs_states_list[i]) # num of data points in a data row
      for t in range(T):
        tmp_alpha_row = []
        for j in range(K):
          if t == 0:
            p = pi[j][0] * b[j][obs_index[obs_states_list[i][t]]]
            tmp_alpha_row.append(p)
            continue
          sum_a_alpha = 0
          for k in range(K):
            sum_a_alpha += a[k][j] * alpha[-1][k]
          p = b[j][obs_index[obs_states_list[i][t]]] * sum_a_alpha
          tmp_alpha_row.append(p)
        alpha.append(tmp_alpha_row)

      log_likelihood += np.log(sum(alpha[-1]))

      # backward
      beta = []
      for t in range(T - 1, -1, -1):
        tmp_beta_row = []
        for j in range(K):
          if t == T - 1:
            tmp_beta_row.append(1)
            continue
          sum_b_beta_a = 0
          for k in range(K):
            sum_b_beta_a += b[k][obs_index[obs_states_list[i][t + 1]]] * beta[0][k] * a[j][k]
          tmp_beta_row.append(sum_b_beta_a)
        beta.insert(0, tmp_beta_row)
      
      # predict y_hat_t
      p_yt_given_x1toT = np.multiply(np.array(alpha), np.array(beta))
      p_yt_given_x1toT = p_yt_given_x1toT.tolist()
      y_hat = []
      for t in range(len(obs_states_list[i])):
        hid_state_t = p_yt_given_x1toT[t].index(max(p_yt_given_x1toT[t]))
        y_hat.append(hid_state_t)
        predict_str += f"{obs_states_list[i][t]}_{index_hid[hid_state_t]} "
      predict_str = predict_str[:-1] + '\n'
      predictions.append(y_hat)

    f_out = open(predict_file,"w+")
    f_out.write(predict_str)
    
    accuracy_count = 0
    total_hid_state_count = 0
    for i in range(N):
      for t in range(len(hid_states_list[i])):
        if hid_index[hid_states_list[i][t]] == predictions[i][t]:
          accuracy_count += 1
        total_hid_state_count += 1
    
    metric_str = f"Average Log-Likelihood: {log_likelihood / N}\n"
    metric_str += f"Accuracy: {accuracy_count / total_hid_state_count}\n"
    f_out = open(metric_file,"w+")
    f_out.write(metric_str)


if __name__ == '__main__':
  test_input = sys.argv[1]
  index_to_word = sys.argv[2]
  index_to_tag = sys.argv[3]
  hmmprior = sys.argv[4]
  hmmemit = sys.argv[5]
  hmmtrans = sys.argv[6]
  predicted_file = sys.argv[7]
  metric_file = sys.argv[8]
  
  hmm = HiddenMarkovModelTester()
  hmm.test(test_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file)
