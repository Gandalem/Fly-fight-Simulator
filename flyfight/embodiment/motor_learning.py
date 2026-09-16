"""Small, local actor/critic readout; these are NOT MaleCNS synapses.

Only descending-neuron activity enters this adapter. Gaussian exploration and
reward-modulated eligibility change continuous motor outputs, without a combat
state machine, demonstrations, backpropagation through the CNS, or a replay buffer.
"""
import copy
import numpy as np


ACTION_NAMES = (
    'walk_left', 'walk_right',
    'lf_coxa_pitch', 'lf_coxa_roll', 'lf_tibia_pitch',
    'rf_coxa_pitch', 'rf_coxa_roll', 'rf_tibia_pitch',
    'lf_adhesion', 'rf_adhesion',
)


class MotorLearningState:
    ARRAY_NAMES = ('weights', 'value_weights', 'actor_trace', 'value_trace',
                   'filtered', 'previous_features', 'score', 'action')

    def __init__(self, seed, config):
        self.config = copy.deepcopy(config['motor_learning'])
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 7301]))
        n = self.config['feature_groups']
        self.weights = np.zeros((len(ACTION_NAMES), n+1), np.float32)
        self.value_weights = np.zeros(n+1, np.float32)
        self.actor_trace = np.zeros_like(self.weights)
        self.value_trace = np.zeros_like(self.value_weights)
        self.filtered = np.zeros(n, np.float32)
        self.previous_features = np.zeros(n+1, np.float32)
        self.score = np.zeros_like(self.weights)
        self.action = np.zeros(len(ACTION_NAMES), np.float32)
        self.decisions = self.updates = 0
        self.last_td_error = 0.
        self.reset('A')

    def reset(self, mode='B', decay=.8):
        if mode == 'A':
            self.weights.fill(0)
            self.value_weights.fill(0)
            self.decisions = self.updates = 0
        elif mode == 'C':
            self.weights *= decay
            self.value_weights *= decay
        elif mode != 'B':
            raise ValueError(mode)
        # Teleporting a restored body is not an action transition. Clear only the
        # short motor trajectory; CNS eligibility and learned weights are separate.
        for name in self.ARRAY_NAMES[2:]:
            getattr(self, name).fill(0)
        self.pending = False
        self.elapsed = self.reward = 0.
        self.last_td_error = 0.

    def _features(self):
        return np.r_[self.filtered/max(1.,np.linalg.norm(self.filtered)), 1.].astype(np.float32)

    def mean_action(self, features):
        mean = self.weights @ features
        # Neutral actuator priors: forward walking and attached front feet.
        # No prior depends on an opponent, food location, damage, or time.
        mean[:2] += self.config['initial_walk_bias']
        mean[8:] += self.config['initial_adhesion_bias']
        return mean

    def decide(self, rates, dt, *, explore=True, learn=True):
        decay = np.exp(-dt/self.config['feature_tau'])
        self.filtered *= decay
        self.filtered += (1-decay)*np.asarray(rates, dtype=np.float32)
        features = self._features()
        if self.pending and self.elapsed+1e-9 < self.config['decision_dt']:
            return self.action.copy()
        if self.pending:
            self._update(features, learn=learn, terminal=False)
        mean = self.mean_action(features)
        noise = self.rng.normal(0, self.config['exploration_std'], len(ACTION_NAMES)) if explore else np.zeros(len(ACTION_NAMES))
        self.action[:] = np.tanh(mean+noise)
        # Score of the sampled latent Gaussian. The tanh Jacobian does not
        # depend on the mean when differentiating log pi for a fixed action.
        self.score[:] = np.outer(noise/self.config['exploration_std']**2, features)
        self.previous_features[:] = features
        self.pending = True
        self.elapsed = self.reward = 0.
        self.decisions += 1
        return self.action.copy()

    def observe(self, reward, dt):
        if self.pending:
            self.reward += np.exp(-self.elapsed/self.config['discount_tau'])*reward
            self.elapsed += dt

    def _update(self, next_features, *, learn, terminal):
        c = self.config
        gamma = np.exp(-self.elapsed/c['discount_tau'])
        decay = gamma*np.exp(-self.elapsed/c['eligibility_tau'])
        value = float(self.value_weights @ self.previous_features)
        following = 0. if terminal else float(self.value_weights @ next_features)
        error = float(np.clip(self.reward+gamma*following-value, -c['td_clip'], c['td_clip']))
        self.last_td_error = error
        self.actor_trace *= decay
        self.actor_trace += self.score
        self.value_trace *= decay
        self.value_trace += self.previous_features
        np.clip(self.actor_trace, -c['trace_clip'], c['trace_clip'], out=self.actor_trace)
        np.clip(self.value_trace, -c['trace_clip'], c['trace_clip'], out=self.value_trace)
        if learn:
            self.weights += c['actor_rate']*error*self.actor_trace
            self.value_weights += c['critic_rate']*error*self.value_trace
            np.clip(self.weights, -c['weight_bound'], c['weight_bound'], out=self.weights)
            np.clip(self.value_weights, -c['weight_bound'], c['weight_bound'], out=self.value_weights)
            self.updates += 1

    def finish(self, *, learn=True):
        if self.pending:
            self._update(self._features(), learn=learn, terminal=True)
        self.pending = False
        self.elapsed = self.reward = 0.

    def metadata(self):
        return dict(rng=self.rng.bit_generator.state, decisions=self.decisions,
                    updates=self.updates, pending=self.pending, elapsed=self.elapsed,
                    reward=self.reward, last_td_error=self.last_td_error, config=self.config)

    def restore_metadata(self, meta):
        self.rng.bit_generator.state = meta['rng']
        for key in ('decisions', 'updates', 'pending', 'elapsed', 'reward', 'last_td_error'):
            setattr(self, key, meta[key])

    def statistics(self):
        return dict(kind='engineering_motor_readout', decisions=self.decisions,
                    updates=self.updates, parameters=int(self.weights.size+self.value_weights.size),
                    changed_actor_weights=int(np.count_nonzero(self.weights)),
                    actor_weight_norm=float(np.linalg.norm(self.weights)),
                    last_td_error=self.last_td_error)

    @property
    def nbytes(self):
        return sum(getattr(self, name).nbytes for name in self.ARRAY_NAMES)
