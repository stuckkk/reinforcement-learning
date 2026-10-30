import util, random

class Agent:

  def getAction(self, state):
    """
    For the given state, get the agent's chosen
    action.  The agent knows the legal actions
    """
    abstract

  def getValue(self, state):
    """
    Get the value of the state.
    """
    abstract

  def getQValue(self, state, action):
    """
    Get the q-value of the state action pair.
    """
    abstract

  def getPolicy(self, state):
    """
    Get the policy recommendation for the state.

    May or may not be the same as "getAction".
    """
    abstract

  def update(self, state, action, nextState, reward):
    """
    Update the internal state of a learning agent
    according to the (state, action, nextState)
    transistion and the given reward.
    """
    abstract


class RandomAgent(Agent):
  """
  Clueless random agent, used only for testing.
  """

  def __init__(self, actionFunction):
    self.actionFunction = actionFunction

  def getAction(self, state):
    return random.choice(self.actionFunction(state))

  def getValue(self, state):
    return 0.0

  def getQValue(self, state, action):
    return 0.0

  def getPolicy(self, state):
    return 'random'

  def update(self, state, action, nextState, reward):
    pass


################################################################################
# Exercise 2

class ValueIterationAgent(Agent):

  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations

    self.value_function = util.Counter()

    for _ in range(self.iterations):
      for state in self.mdp.getStates():
        reward_values_per_action = []

        for action in self.mdp.getPossibleActions(state):
          immediate_reward = self.mdp.getReward(state, action, None)
          future_estimated_reward = sum([self.discount * prob * self.value_function.getCount(next_state) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])
          reward_values_per_action.append(immediate_reward + future_estimated_reward)
        if reward_values_per_action:
          self.value_function[state] = max(reward_values_per_action)


  def getValue(self, state):
    """
    Look up the value of the state (after the indicated
    number of value iteration passes).
    """
    return self.value_function.getCount(state)


  def getQValue(self, state, action):
    """
    Look up the q-value of the state action pair
    (after the indicated number of value iteration
    passes).  Note that value iteration does not
    necessarily create this quantity and you may have
    to derive it on the fly.
    """
    return self.mdp.getReward(state, action, None) + self.discount * sum([prob * self.getValue(next_state) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])


  def getPolicy(self, state):
    """
    Look up the policy's recommendation for the state
    (after the indicated number of value iteration passes).
    """
    best_known_reward = 0
    best_action = None
    for action in self.mdp.getPossibleActions(state):
        reward = self.mdp.getReward(state, action, None) + self.discount * sum([prob * self.getValue(next_state) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])
        if not best_action or reward > best_known_reward:
          best_action = action
          best_known_reward = reward
    return best_action
    

  def getAction(self, state):
    """
    Return the action recommended by the policy.
    """
    return self.getPolicy(state)


  def update(self, state, action, nextState, reward):
    """
    Not used for value iteration agents!
    """

    pass


class PolicyIterationAgent(Agent):
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.policy = {}

    # Initialize policy with any action
    for state in self.mdp.getStates():
      actions = self.mdp.getPossibleActions(state)
      if actions:
        self.policy[state] = actions[0]


    self.value_function = util.Counter()

    for i in range(self.iterations):
      if not -1e-10 < self.value_function.getCount(self.mdp.getStartState()) < 1e-10:
        print(f"Start state value function non-zero at iteration {i}")
      # Policy evaluation
      for _ in range(self.iterations):
        for state in self.mdp.getStates():
          # Do not change the value function if no action is available
          if self.mdp.getPossibleActions(state):
            action = self.getAction(state)
            expected_future_reward = self.discount * sum([prob * self.getValue(next_state) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])
            self.value_function[state] = self.mdp.getReward(state, action, None) + expected_future_reward

      # Policy iteration
      policy_stable = True
      for state in self.mdp.getStates():
        best_action = None
        best_reward = 0
        for action in self.mdp.getPossibleActions(state):
          expected_future_reward = self.discount * sum([prob * self.getValue(next_state) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])
          immediate_reward = self.mdp.getReward(state, action, None)
          reward = immediate_reward + expected_future_reward
          if not best_action or reward > best_reward:
            best_action = action
            best_reward = reward

        # Policy changed if there are actions available and new argmax differs from old argmax
        if best_action and best_action != self.getAction(state):
          policy_stable = False
        
        self.policy[state] = best_action

      # Terminate if policy is stable
      if policy_stable:
        print(f"Stable policy after {i} iterations")
        break
        

  def getValue(self, state):
    """
    Look up the value of the state (after the indicated
    number of value iteration passes).
    """
    return self.value_function.getCount(state)


  def getQValue(self, state, action):
    """
    Look up the q-value of the state action pair
    (after the indicated number of value iteration
    passes).  Note that value iteration does not
    necessarily create this quantity and you may have
    to derive it on the fly.
    """
    return self.mdp.getReward(state, action, None) + self.discount * sum([prob * self.getValue(next_state) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])


  def getPolicy(self, state):
    """
    Look up the policy's recommendation for the state
    (after the indicated number of value iteration passes).
    """
    return self.policy[state]
    

  def getAction(self, state):
    """
    Return the action recommended by the policy.
    """
    return self.getPolicy(state)


  def update(self, state, action, nextState, reward):
    """
    Not used for policy iteration agents!
    """

    pass


################################################################################
# Below can be ignored for Exercise 2

class QLearningAgent(Agent):

  def __init__(self, actionFunction, discount = 0.9, learningRate = 0.1, epsilon = 0.2):
    """
    A Q-Learning agent gets nothing about the mdp on
    construction other than a function mapping states to actions.
    The other parameters govern its exploration
    strategy and learning rate.
    """
    self.setLearningRate(learningRate)
    self.setEpsilon(epsilon)
    self.setDiscount(discount)
    self.actionFunction = actionFunction

    raise "Your code here."




  # THESE NEXT METHODS ARE NEEDED TO WIRE YOUR AGENT UP TO THE CRAWLER GUI

  def setLearningRate(self, learningRate):
    self.learningRate = learningRate

  def setEpsilon(self, epsilon):
    self.epsilon = epsilon

  def setDiscount(self, discount):
    self.discount = discount

  # GENERAL RL AGENT METHODS

  def getValue(self, state):
    """
    Look up the current value of the state.
    """

    raise ValueError("Your code here.")



  def getQValue(self, state, action):
    """
    Look up the current q-value of the state action pair.
    """

    raise ValueError("Your code here.")



  def getPolicy(self, state):
    """
    Look up the current recommendation for the state.
    """

    raise ValueError("Your code here.")



  def getAction(self, state):
    """
    Choose an action: this will require that your agent balance
    exploration and exploitation as appropriate.
    """

    raise ValueError("Your code here.")



  def update(self, state, action, nextState, reward):
    """
    Update parameters in response to the observed transition.
    """

    raise ValueError("Your code here.")
