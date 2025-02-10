from .entities import Player, Pipes


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
                return (self.player.cx - pipe.x, self.player.cy - pipe.h)
        return None

    # let's choose an action based on the Q-table
    def pick_action(self):
        state = self.current_state()
        self.q.setdefault(state, {ACTION_FLAP: 0, ACTION_NOP: 0})
        qa = self.q[state]
        if qa[ACTION_FLAP] > qa[ACTION_NOP]:
            return ACTION_FLAP
        return ACTION_NOP

    def tick(self):
        return self.current_state(), self.pick_action()

    def reward(self, state, action, alive: bool):
        next_state = self.current_state()
        # Q-learning update formula: Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
        alpha = 0.1  # learning rate
        gamma = 0.9  # discount factor
        reward = 1 if alive else -1000
        # Get max Q value for next state
        self.q.setdefault(next_state, {ACTION_FLAP: 0, ACTION_NOP: 0})
        next_max_q = max(self.q[next_state].values())
        # Update Q value
        self.q.setdefault(state, {ACTION_FLAP: 0, ACTION_NOP: 0})
        current_q = self.q[state][action]
        self.q[state][action] = current_q + alpha * (reward + gamma * next_max_q - current_q)