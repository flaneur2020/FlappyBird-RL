from .entities import Player, Pipes
import random

ACTION_FLAP = "flap"
ACTION_NOP = "nop"


class Agent:
    def __init__(self):
        self.player = None
        self.pipes = None
        # (player_relative_x, player_relative_y) -> action -> reward
        self.q = {}

    def reset_state(self, player: Player, pipes: Pipes):
        self.player = player
        self.pipes = pipes

    # let's define a key for the Q-table. the state is the playe's
    # relative positions to the next lower pipe.
    def current_state(self):
        for pipe in self.pipes.lower:
            if pipe.cx >= self.player.cx:
                x = int(self.player.cx - pipe.x)
                y = int(self.player.cy - pipe.y)
                return (x - x % 2, y - y % 2)
        return None

    # let's choose an action based on the Q-table
    def pick_action(self):
        state = self.current_state()
        self.q.setdefault(state, {ACTION_FLAP: 0, ACTION_NOP: 0})
        qa = self.q[state]
        if qa[ACTION_FLAP] == qa[ACTION_NOP] and qa[ACTION_FLAP] != 0:
            return random.choice([ACTION_FLAP, ACTION_NOP])
        if qa[ACTION_FLAP] > qa[ACTION_NOP]:
            return ACTION_FLAP
        return ACTION_NOP

    def tick(self):
        return self.current_state(), self.pick_action()

    def reward(self, state, action, alive: bool):
        if state is None or action is None:
            return
        next_state = self.current_state()
        # Q-learning update formula: Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
        alpha = 0.7  # learning rate
        gamma = 0.99  # discount factor
        reward = 1 if alive else -1000
        # Get max Q value for next state
        self.q.setdefault(next_state, {ACTION_FLAP: 0, ACTION_NOP: 0})
        next_max_q = max(self.q[next_state].values())
        # Update Q value
        self.q.setdefault(state, {ACTION_FLAP: 0, ACTION_NOP: 0})
        current_q = self.q[state][action]
        self.q[state][action] = current_q + alpha * (reward + gamma * next_max_q - current_q)
