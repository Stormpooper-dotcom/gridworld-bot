from env import GridworldEnv
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    use_traps = sys.argv[2].lower() == "true"
    # Create and validate the environment format
    env = GridworldEnv(size=10, num_traps=10, use_traps=use_traps)
    print("Checking environment compatibility with Gymnasium...")
    check_env(env, warn=True)
    print("Environment check passed!")

    # Set up TensorBoard logging directory
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Initialise the PPO Model
    # Using 'MlpPolicy' since the observation is a flat vector [dx, dy]
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        n_steps=512,
        batch_size=32,
        n_epochs=4,
        gamma=0.95,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir
    )


    # Train the agent
    TOTAL_TIMESTEPS = int(sys.argv[3])
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("Training finished!")

    # Save the trained model
    model_path = sys.argv[1]
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate the trained agent
    print("Evaluating trained model performance...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Demonstration Loop (Watch the trained model play 3 episodes)
    print("\nRunning demonstration episodes:")
    for episode in range(3):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_episode_reward = 0
        
        while not done:
            # Predict action using deterministic policy for best performance
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            total_episode_reward += reward
            
        print(f" -> Episode {episode + 1}: Finished in {step_count} steps. Total Reward: {total_episode_reward:.2f}")