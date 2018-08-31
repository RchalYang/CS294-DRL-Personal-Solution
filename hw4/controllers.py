import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """

        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high
        self.action_min = env.action_space.low
        self.action_range = self.action_max-self.action_min

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        action = np.random.randn(self.action_dim)
        action = action * self.action_range
        action = action + self.action_min
        
        return action


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.random_sampler = RandomController(env)

        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high
        self.action_min = env.action_space.low
        self.action_range = self.action_max-self.action_min

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """

        import copy

        first_acts = []
        costs = []

        #[TODO] Batch simulation
        # ob = np.repeat(state, self.num_simulated_paths, axis = 0)
        # act = np.random.randn( self.num_simulated_paths, self.action_dim )

        for num_path in range(self.num_simulated_paths):

            ob = copy.deepcopy(state)
            act = self.random_sampler.get_action(ob)
            new_ob = self.dyn_model.predict(act)

            states=[ob]
            next_states=[new_ob]
            actions=[act]

            first_acts.append(act)

            ob = new_ob

            for num_step in range(self.horizon - 1):

                act = self.random_sampler.get_action(ob)
                new_ob = self.dyn_model.predict(act)

                states.append(ob)
                next_states.append(new_ob)
                actions.append(act)

                ob = new_ob

            cost = trajectory_cost_fn(self.cost_fn, states, actions, next_states)
            costs.append(cost)

        max_index = costs.index(max(costs))
        
        return first_acts[max_index]
