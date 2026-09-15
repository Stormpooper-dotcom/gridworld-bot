from env import GridworldEnv

env = GridworldEnv(size=10)
obs, _ = env.reset()
for _ in range(200):
    obs, reward, term, trunc, _ = env.step(env.action_space.sample())
    if term:
        print("Goal reached")
        break