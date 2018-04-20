class Env:
    def __init__(self):
        self.current_state = None

    def reset(self, state):
        self.current_state = state

    def action(self, action):
        pass
