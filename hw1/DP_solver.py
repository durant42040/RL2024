
import numpy as np

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        next_state, reward, done = self.grid_world.step(state, action)
        q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)

        return q_value


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount_factor (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy


    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        value = 0

        for action in range(self.grid_world.get_action_space()):
            value += self.policy[state, action] * self.get_q_value(state, action)

        return value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            values[state] = self.get_state_value(state)

        error = np.max(np.abs(self.values - values))
        self.values = values

        return error


    def run(self) -> None:
        """Run the algorithm until convergence."""
        while True:
            error = self.evaluate()
            if error < self.threshold:
                break



class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        return self.get_q_value(state, self.policy[state])

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            values[state] = self.get_state_value(state)

        error = np.max(np.abs(self.values - values))
        self.values = values

        return error

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        stable = True

        policy = np.zeros((self.grid_world.get_state_space(), self.grid_world.get_action_space()))

        for state in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(q_values)
            policy[state, best_action] = 1

            if best_action != self.policy[state]:
                stable = False

        self.policy = np.argmax(policy, axis=1)

        return stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        while True:
            while True:
                error = self.policy_evaluation()
                if error < self.threshold:
                    break

            stable = self.policy_improvement()
            if stable:
                break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        return np.max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            values[state] = self.get_state_value(state)

        error = np.max(np.abs(self.values - values))
        self.values = values

        return error


    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        for state in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(q_values)
            self.policy[state] = best_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        while True:
            error = self.policy_evaluation()
            if error < self.threshold:
                break

        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def in_place_dp(self):
        """Perform the in-place dynamic programming update"""
        while True:
            error = 0
            for state in range(self.grid_world.get_state_space()):
                value = np.max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
                error = max(error, np.abs(self.values[state] - value))
                self.values[state] = value
            if error < self.threshold:
                break

        for state in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(q_values)
            self.policy[state] = best_action

    def prioritized_sweeping(self):
        """Perform the prioritized sweeping update"""
        predecessors = [set() for _ in range(self.grid_world.get_state_space())]
        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                next_state, _, _ = self.grid_world.step(state, action)
                if next_state != state:
                    predecessors[next_state].add(state)

        queue = []
        for state in range(self.grid_world.get_state_space()):
            updated_value = np.max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
            bellman_error = np.abs(self.values[state] - updated_value)
            if bellman_error > self.threshold:
                queue.append((state, updated_value, bellman_error))

        while queue:
            state, updated_value, bellman_error = max(queue, key=lambda x: x[2])
            queue.remove((state, updated_value, bellman_error))

            self.values[state] = updated_value

            for predecessor in predecessors[state]:
                updated_value = np.max([self.get_q_value(predecessor, action) for action in range(self.grid_world.get_action_space())])
                bellman_error = np.abs(self.values[predecessor] - updated_value)
                if bellman_error > self.threshold:
                    if any(predecessor == item[0] for item in queue):
                        queue = [item for item in queue if item[0] != predecessor]
                    queue.append((predecessor, updated_value, bellman_error))

        for state in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(q_values)
            self.policy[state] = best_action


    def fixed_step_value_iteration(self):
        for _ in range(int(self.grid_world.get_state_space() / 2) - 2):
            for state in range(self.grid_world.get_state_space()):
                q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
                self.values[state] = np.max(q_values)

        for state in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            best_action = np.argmax(q_values)
            self.policy[state] = best_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        # self.in_place_dp()
        # self.prioritized_sweeping()
        self.fixed_step_value_iteration()

