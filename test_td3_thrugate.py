
def test(env, agent, render=True, max_t_step=1000, explore=False, n_times=1):
    sum_scores = 0
    for i in range(n_times):
        state = env.reset()
        score = 0
        done=False
        t = int(0)
        while not done and t < max_t_step:
            t += int(1)
            action = agent.get_action(state, explore=explore)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            #print(action)
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward
            if render:
                env.render()
        sum_scores += score
    mean_score = sum_scores/n_times
    print('\rTest Episodes\tMean Score: {:.2f}'.format(mean_score))
    return mean_score