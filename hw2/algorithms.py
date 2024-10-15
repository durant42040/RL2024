import numpy as np
from gridworld import GridWorld


# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """

    def __init__(
        self,
        grid_world: GridWorld,
        policy: np.ndarray = None,
        discount_factor: float = 1.0,
        max_episode: int = 300,
        seed: int = 1,
    ):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0
        self.action_space = grid_world.get_action_space()
        self.state_space = grid_world.get_state_space()
        self.values = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)  # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = (
                np.ones((self.state_space, self.action_space)) / self.action_space
            )  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state

        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]
        action = self.rng.choice(self.action_space, p=action_probs)

        next_state, reward, done = self.grid_world.step(action)
        if done:
            self.episode_counter += 1
        return next_state, reward, done


class MonteCarloPrediction(ModelFreePrediction):
    def __init__(
        self,
        grid_world: GridWorld,
        policy: np.ndarray = None,
        discount_factor: float = 1.0,
        max_episode: int = 300,
        seed: int = 1,
    ):
        """
        Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        returns = [[] for _ in range(self.state_space)]
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            episode = []
            G = 0
            while True:
                next_state, reward, done = self.collect_data()
                episode.append((current_state, reward))
                current_state = next_state
                if done:
                    break

            while episode:
                state, reward = episode.pop()
                G = self.discount_factor * G + reward

                if any([state == e[0] for e in episode]):
                    continue

                returns[state].append(G)
                self.values[state] = np.mean(returns[state])


class TDPrediction(ModelFreePrediction):
    def __init__(
        self,
        grid_world: GridWorld,
        learning_rate: float,
        policy: np.ndarray = None,
        discount_factor: float = 1.0,
        max_episode: int = 300,
        seed: int = 1,
    ):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)
        self.lr = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            done = False
            while not done:
                next_state, reward, done = self.collect_data()

                self.values[current_state] += self.lr * (
                    reward
                    + self.discount_factor * (1 - done) * self.values[next_state]
                    - self.values[current_state]
                )
                current_state = next_state


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
        self,
        grid_world: GridWorld,
        learning_rate: float,
        num_step: int,
        policy: np.ndarray = None,
        discount_factor: float = 1.0,
        max_episode: int = 300,
        seed: int = 1,
    ):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)
        self.lr = learning_rate
        self.n = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            T = np.inf
            t = 0
            episode = {0: (current_state, 0)}
            while True:
                if t < T:
                    next_state, reward, done = self.collect_data()
                    episode[t + 1] = (next_state, reward)
                    if done:
                        T = t + 1

                tau = t - self.n + 1

                if tau >= 0:
                    G = sum(
                        [
                            self.discount_factor ** (i - tau - 1) * episode[i][1]
                            for i in range(tau + 1, min(tau + self.n, T) + 1)
                        ]
                    )

                    if tau + self.n < T:
                        G += (
                            self.discount_factor**self.n
                            * self.values[episode[tau + self.n][0]]
                        )
                    self.values[episode[tau][0]] += self.lr * (
                        G - self.values[episode[tau][0]]
                    )
                t += 1
                if tau == T - 1:
                    break


# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space = grid_world.get_state_space()
        self.q_values = np.zeros((self.state_space, self.action_space))
        self.policy = (
            np.ones((self.state_space, self.action_space)) / self.action_space
        )  # stochastic policy
        self.policy_index = np.zeros(
            self.state_space, dtype=int
        )  # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index

    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values


class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
        self,
        grid_world: GridWorld,
        discount_factor: float,
        learning_rate: float,
        epsilon: float,
        seed: int = 1,
    ):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        for t, (state, action, reward) in enumerate(
            zip(state_trace, action_trace, reward_trace)
        ):
            G = 0
            for i in range(t, len(reward_trace)):
                G += self.discount_factor ** (i - t) * reward_trace[i]
            self.q_values[state, action] += self.lr * (G - self.q_values[state, action])

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        max_values = self.get_max_state_values()
        for state in range(self.state_space):
            self.policy[state] = np.zeros(self.action_space)
            n = np.count_nonzero(self.q_values[state] == max_values[state])
            for action in range(self.action_space):
                if self.q_values[state, action] == max_values[state]:
                    self.policy[state][action] = (1 - self.epsilon) / n
                self.policy[state, action] += self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace = [current_state]
        action_trace = []
        reward_trace = []

        while iter_episode < max_episode:
            iter_episode += 1
            done = False
            while not done:
                action = self.rng.choice(
                    self.action_space, p=self.policy[current_state]
                )

                next_state, reward, done = self.grid_world.step(action)
                action_trace.append(action)
                reward_trace.append(reward)
                if not done:
                    state_trace.append(next_state)
                current_state = next_state

            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()
            state_trace = [current_state]
            action_trace = []
            reward_trace = []


class SARSA(ModelFreeControl):
    def __init__(
        self,
        grid_world: GridWorld,
        discount_factor: float,
        learning_rate: float,
        epsilon: float,
        seed: int = 1,
    ):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        self.q_values[s, a] += self.lr * (
            r
            + self.discount_factor * (1 - is_done) * self.q_values[s2, a2]
            - self.q_values[s, a]
        )

        self.policy[s] = np.zeros(self.action_space)
        max_value = self.q_values[s].max()

        n = np.count_nonzero(self.q_values[s] == max_value)
        for action in range(self.action_space):
            if self.q_values[s, action] == max_value:
                self.policy[s, action] = (1 - self.epsilon) / n
            self.policy[s, action] += self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        iter_episode = 0
        current_state = self.grid_world.reset()

        while iter_episode < max_episode:
            action = self.rng.choice(self.action_space, p=self.policy[current_state])
            is_done = False
            reward_trace = []
            while not is_done:
                next_state, reward, is_done = self.grid_world.step(action)
                reward_trace.append(reward)
                next_action = self.rng.choice(
                    self.action_space, p=self.policy[next_state]
                )

                self.policy_eval_improve(
                    current_state, action, reward, next_state, next_action, is_done
                )
                current_state = next_state
                action = next_action

            iter_episode += 1


class Q_Learning(ModelFreeControl):
    def __init__(
        self,
        grid_world: GridWorld,
        discount_factor: float,
        learning_rate: float,
        epsilon: float,
        buffer_size: int,
        update_frequency: int,
        sample_batch_size: int,
        seed: int = 1,
    ):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.buffer = []
        self.update_frequency = update_frequency
        self.sample_batch_size = sample_batch_size
        self.rng = np.random.default_rng(seed=seed)

    def add_buffer(self, s, a, r, s2, d) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        if len(self.buffer) < self.sample_batch_size:
            return np.array(self.buffer)

        buffer = np.array(self.buffer)
        indices = np.random.choice(
            buffer.shape[0], self.sample_batch_size, replace=False
        )
        return buffer[indices]

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        s, a, r, s2 = int(s), int(a), r, int(s2)
        self.q_values[s, a] += self.lr * (
            r
            + self.discount_factor * (1 - is_done) * self.q_values[s2].max()
            - self.q_values[s, a]
        )

        self.policy[s] = np.zeros(self.action_space)
        max_value = self.q_values[s].max()

        n = np.count_nonzero(self.q_values[s] == max_value)
        for action in range(self.action_space):
            if self.q_values[s, action] == max_value:
                self.policy[s, action] = (1 - self.epsilon) / n
            self.policy[s, action] += self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        iter_episode = 0
        current_state = self.grid_world.reset()
        transition_count = 0

        while iter_episode < max_episode:
            is_done = False
            reward_trace = []
            while not is_done:
                action = self.rng.choice(
                    self.action_space, p=self.policy[current_state]
                )
                next_state, reward, is_done = self.grid_world.step(action)
                reward_trace.append(reward)

                self.add_buffer(current_state, action, reward, next_state, is_done)

                transition_count += 1

                if transition_count % self.update_frequency == 0:
                    B = self.sample_batch()

                    for s, a, r, s2, d in B:
                        self.policy_eval_improve(s, a, r, s2, d)

                current_state = next_state

            iter_episode += 1
