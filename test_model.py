import time

from Game import Game
from Agent import Agent
def main():
    env = Game()

    showcases = 10

    episode_rewards = []

    input_dimensions = env.get_state().shape
    model = "model362018_3"
    agent = Agent(input_dimensions, env.num_actions, model_path=model)

    for i in range(showcases):
        done = False
        state = env.reset()
        while not done:
            action = agent.get_action(state)

            new_state, reward, done, extra_info = env.step(action)

            env.render()
            time.sleep(0.1)
            state = new_state

    env.close()
    return

if __name__ == "__main__":
    main()

