import numpy as np
import tensorflow as tf
import argparse
import datetime

from cnn import CNN
from net_manager import NetManager
from reinforce import Reinforce

from tensorflow.examples.tutorials.mnist import input_data

def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_layers', help="max layer of cnn", default=2, type=int)
    parser.add_argument('--mode', help="mode of nas, current support modes: [train, evaluate]", default='train')
    parser.add_argument('--action', help="specified action. In train mode, it is the initial action; In evaluate mode, it is the evaluted action",
                            default=None)
    parser.add_argument('--evaluate_time', help="evaluate time of evaluate mode", default=1, type=int)

    args = parser.parse_args()
    #args.max_layers = int(args.max_layers)
    return args


'''
    Policy network is a main network for searching optimal architecture
    it uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1363

    Args:
        state: current state of required topology
        max_layers: maximum number of layers
    Returns:
        3-D tensor with new state (new topology)
'''
def policy_network(state, max_layers):
    with tf.name_scope("policy_network"):
        nas_cell = tf.contrib.rnn.NASCell(4*max_layers)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )
        bias = tf.Variable([0.05]*4*max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        print("outputs: ", outputs, outputs[:, -1:, :],  tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]))
        #return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn
        return outputs[:, -1:, :]      

def train(dataset, learning_rate=0.001, batch_size=100, num_input=784, num_classes=10, train_size=100000, test_size=10000):
    global args
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate_op = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_op)

    reinforce = Reinforce(sess, optimizer, policy_network, args.max_layers, global_step)
    net_manager = NetManager(num_input=num_input,
                             num_classes=num_classes,
                             learning_rate=learning_rate,
                             dataset=dataset,
                             bathc_size=batch_size,
                             train_size=train_size,
                             test_size=test_size)

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0]*args.max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0
    max_acc = 0.0
    max_action = None
    for i_episode in range(MAX_EPISODES):       
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
            print("=====>", reward, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        reinforce.storeRollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log_max_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" max accuracy:  "+str(max_acc)+" action:  "+str(max_action)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.write(log_max_str)
        log.close()
        print(log_str)
        print(log_max_str)

def main():
    global args
    args = parse_args()
    print(args)


    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #train(mnist)

if __name__ == '__main__':
  main()
