import time


class RunningSum(object):
    def __init__(self, decay_rate, init=None):
        self._decay_rate = decay_rate
        self._keep_rate = 1 - decay_rate
        self._value = None

    def update(self, x):
        if self._value is None:
            self._value = x
        else:
            self._value = self._decay_rate * self._value + self._keep_rate * x
        return self._value

    def rate(self):
        return self._decay_rate

    def get(self):
        return self._value


class DiffTrainer(object):
    """An agent trainer that uses state different as training data."""
    def __init__(self, batch_size, prepro):
        self._batch_size = batch_size
        self._prepro = prepro

    def play(self, agent, env):
        init = self._prepro(env.reset())
        pre_state = init
        cur_state = init
        total_rewards = 0
        step = 0
        done = False
        while not done:
            step += 1
            x = cur_state - pre_state
            action = agent.act(x)
            pre_state = cur_state
            observation, reward, done, _info = env.step(action + 1)
            cur_state = self._prepro(observation)
            agent.percept(x, action, reward)
            total_rewards += reward
            if done:
                break
        env.close()
        return total_rewards, step

    def train(self, agent, env, episodes):
        running_rewards = [RunningSum(1 - x) for x in [1e-1, 1e-2, 1e-3]]

        t0 = time.time()
        for i in range(episodes):
            t1 = time.time()
            tot_reward, step = self.play(agent, env)
            for r in running_rewards:
                r.update(tot_reward)
            now = time.time()
            dur = now - t1
            total_dur = now - t0

            runnings = ', '.join('%s-running reward: %-8.2f' %
                                 (r.rate(), r.get()) for r in running_rewards)

            print(
                'episode: %-3d end in %d steps, took %6.2fs (%6.2fms/step), reward: %8.2f; all took %8.2fs; %s'
                % (
                    i + 1,
                    step,
                    dur,
                    1000.0 * dur / step,
                    tot_reward,
                    total_dur,
                    runnings,
                ))
            if (i + 1) % self._batch_size == 0 or (i + 1) == episodes:
                t1 = time.time()
                agent.learn()
                dur = time.time() - t1
                print('learn from history took %8.2fs' % dur)
