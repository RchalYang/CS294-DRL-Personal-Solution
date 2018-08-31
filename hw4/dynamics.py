import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.output_shape = env.observation_space.shape[0]

        self.batch_size    = batch_size
        self.normalization = normalization
        self.iters    = iterations
        self.lr = learning_rate
        self.sess = sess

        self.ob_space  = env.observation_space.shape[0]
        self.act_space = env.actions_space.shape[0]

        self.input_data_placeholder = tf.placeholder(tf.float32, shape= ( self.ob_space + self.act_space ))
        self.output = build_mlp( self.input_data_placeholder , self.output_shape,
                "dynamics", n_layers, size, activation, output_activation )
        
        self.delta_placeholder = tf.placeholder(tf.float32, shape = (self.ob_space) )

        self.dynamic_loss = tf.reduce_mean(tf.square(self.output - self.delta_placeholder))
        # self.optimizer = tf.train.optimizer.
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.dynamic_loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        states, actions, next_states = data
        deltas = next_states - states
        
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = self.normalization

        normed_obs    = ( states  - mean_obs    ) / ( std_obs    + 1e-8 )
        normed_acts   = ( actions - mean_action ) / ( std_action + 1e-8 )
        normed_deltas = ( deltas  - mean_deltas ) / ( std_deltas + 1e-8 )

        total_num = len(states)
        batch_num = (total_num // self.batch_size) if (total_num % self.batch_size == 0) \
            else  (total_num // self.batch_size + 1)
            
        for i in range(batch_num):
            batch_start = i     * self.batch_size
            batch_end   = (i+1) * self.batch_size

            batch_obs    = normed_obs[batch_start : batch_end]
            batch_acts   = normed_acts[batch_start : batch_end]
            batch_deltas = normed_deltas[batch_start: batch_end]

            input_data = np.concatenate( (batch_obs, batch_acts), axis = 1 )

            loss = self.sess.run([self.optimizer], 
                feed_dict = { self.input_data_placeholder: input_data,
                              self.delta_placeholder: batch_deltas }
                )

            # self.sess.run()
            print("Training Loss for fit Dynamics:{}".format(loss))

        """YOUR CODE HERE """


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) statens ad (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = self.normalization

        normed_obs    = ( states  - mean_obs    ) / ( std_obs    + 1e-8 )
        normed_acts   = ( actions - mean_action ) / ( std_action + 1e-8 )
        

        total_num = len(states)
        batch_num = (total_num // self.batch_size) if (total_num % self.batch_size == 0) \
            else  (total_num // self.batch_size + 1)
            
        deltas = []

        for i in range(batch_num):
            batch_start = i     * self.batch_size
            batch_end   = (i+1) * self.batch_size

            batch_obs    = normed_obs[batch_start : batch_end]
            batch_acts   = normed_acts[batch_start : batch_end]

            input_data = np.concatenate( (batch_obs, batch_acts), axis = 1 )

            batch_deltas = self.sess.run(self.output, 
                feed_dict = { self.input_data_placeholder: input_data }
            )

            # self.sess.run()
            deltas.append(batch_deltas)

        normed_deltas = np.concatenate( deltas, axis = 0 )
        deltas = normed_deltas * std_deltas + mean_deltas

        next_states = states + deltas

        return next_states