import sys
from stable_baselines3 import PPO
from env import GridworldEnv

def evaluate_agent(model_path, num_episodes=100):
    # Initialize the exact same environment used for testing
    env = GridworldEnv(size=10, num_traps=3, use_traps=True, max_steps=100)
    
    # Load the trained network
    print(f"Loading model from: {model_path}.zip...")
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you provide the correct filename without the '.zip' extension.")
        return

    # Track metrics
    success_count = 0
    trap_count = 0
    timeout_count = 0
    total_steps = 0

    print(f"\nRunning {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # deterministic=True ensures the agent uses its best learned path instead of exploring
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                total_steps += env.current_step
                
                # Check how the episode ended
                if env.agent == env.goal:
                    success_count += 1
                elif env.use_traps and (env.agent in env.traps):
                    trap_count += 1
                elif truncated or env.current_step >= env.max_steps:
                    timeout_count += 1

    # Calculate percentages
    success_rate = (success_count / num_episodes) * 100
    trap_rate = (trap_count / num_episodes) * 100
    timeout_rate = (timeout_count / num_episodes) * 100
    avg_steps = total_steps / num_episodes if num_episodes > 0 else 0

    # Print a clean report card
    print("\n" + "="*40)
    print("           EVALUATION RESULTS           ")
    print("="*40)
    print(f"Total Episodes Run:  {num_episodes}")
    print(f"Success Rate:        {success_rate:.1f}%  ({success_count}/{num_episodes}) 🏁")
    print(f"Hit a Trap Rate:     {trap_rate:.1f}%  ({trap_count}/{num_episodes}) ❌")
    print(f"Timed Out Rate:      {timeout_rate:.1f}%  ({timeout_count}/{num_episodes}) ⏳")
    print(f"Avg Steps per Win:   {avg_steps:.1f} steps")
    print("="*40)

if __name__ == "__main__":
    # Check if a model name was passed as a command-line argument
    if len(sys.argv) > 1:
        target_model = sys.argv[1]
    else:
        # Default fallback name matching our training setups
        target_model = "ppo_gridworld_mini_brain"
        
    evaluate_agent(target_model, num_episodes=1000)
