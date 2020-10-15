#!/usr/bin/env python3
import random
from Game import Game
import time

from Agent import Agent


def main():
    env = Game()

    episodes = 1000000
    render_period = 100000
    render = True

    episode_rewards = []

    input_dimensions = env.get_state().shape
    agent = Agent(input_dimensions, env.num_actions)

    for episode in range(episodes):
        episode_reward = 0
        state, reward = env.reset()

        while not env.done:
            if agent.epsilon > random.random():
                #preform random action
                #while epsilon is high more random actions will be taken
                action = random.randint(0, env.num_actions-1)
            else:
                #preform action based off network prediction
                #as episilon decays this will be the usual option
                action = agent.get_action(state)

            new_state, reward, done = env.step(action)

            # train
            env_info = (state, action, new_state, reward, done)
            agent.train(env_info)

            # render
            if render and (episode % render_period == 0):
                env.render()
                time.sleep(0.1) #helps view game at normal speed

            state = new_state
            episode_reward += reward
        episode_rewards.append(episode_reward)
        if render and (episode % render_period == 0):
            print(f"episode: {episode} reward: {episode_reward}")
        if episode_reward >= 3:
            agent.model.save(f"./model{episode}_{episode_reward}")

if __name__ == "__main__":
    main()
