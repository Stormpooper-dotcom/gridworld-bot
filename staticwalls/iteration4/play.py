import time
import sys
from stable_baselines3 import PPO
from env import GridworldEnv

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    use_traps = True if sys.argv[2].lower() == "true" else False
    
    # Initialize the environment
    env = GridworldEnv(size=10, num_traps=10, use_traps=use_traps)
    
    print(f"Loading weights from model archive: '{MODEL_PATH}'...")
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print("Model file loaded successfully!\n")
    except FileNotFoundError:
        print(f"Error: Could not find '{MODEL_PATH}'. Run your training script first!")
        exit()

    NUM_EPISODES = int(sys.argv[3])
    ACTION_NAMES = {0: "Up ⬆️", 1: "Down ⬇️", 2: "Left ⬅️", 3: "Right ➡️"}

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\n🎬 STARTING EPISODE {ep + 1}")
        env.render_ascii()
        time.sleep(0.5)
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            step += 1
            total_reward += reward
            
            print(f"Step {step} | Action Taken: {ACTION_NAMES[int(action)]} | Reward received: {reward:.2f}")
            env.render_ascii()
            
            time.sleep(0.2)  # Control game speed
            
        print(f"🎉 Episode {ep + 1} finished! Total moves: {step} | Accumulated reward: {total_reward:.2f}\n")
