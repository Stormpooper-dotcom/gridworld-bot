# gridworld-bot
An investigation into AI training. I built an efficient AI to play Gridworld, and trained it to 91% accuracy in only 150k steps.

Unfortunately, I didn't keep track of my iterations 1, 2, and 3 so they couldn't make it here. However 4 and 5, and `current` (a copy of 5 currently) did make it!

Feel free to train up your own AI using `train.py`, update my `env.py` if you want to try make it more efficient, or just play around with mine using `test_agent.py` and `play.py`. An explanation of these files follows.

### Files
- `env.py`: A Gymnasium-style environment. For those who don't know, the environment basically takes the model's action and updates the game state, giving the model a reward based on this new state. It also gives the model an observation, or list of numbers about what the AI can "see".
- `train.py`: For those of you with a computer newer than 2016. Turn your PC into a heat machine by forcing it to calculate how to walk around a grid properly! Pretty self explanatory really. Use TensorBoard to read the logs from training, you can see mine as well in `tensorboard_logs`.
- `play.py`: By far the most interesting one. Visualise exactly what the AI is actually doing. Takes 3 arguments: the model path, whether or not to use traps, and the number of runs.
- `test_agent.py`: Once you've run `play` you can see the model's statistics with thousands of games. Be aware that this can take a LONG time on models with a bad success rate (like `current/model_traps_new.zip`).

### Gridworld rules
You start in the top left corner of an x-by-y grid of cells. Some have traps, these are generated randomly. Get to the goal.

### Technical stuff
Action space: 4 discrete actions: move Up, Down, Left, or Right. numbered 0 through 3 in the env.

Observation space: in Iteration 5 it is a 10 item array (ignore the comment). 0 and 1 are the distance to the goal, ie dx and dy. 2, 3, 4, 5 are the distances to the walls. 6, 7, 8, 9 are binary: if there is a trap immediately Up, Down, Left or Right of the AI.

As you can see, very simple.

### Conclusion
Always remember: when training AI, ensure you've covered EVERYTHING. Iteration 4 had reward shaping based on the change in distance to the goal, the AI decided to infinitely step back and forth next to the goal to farm reward. Therefore, simple is ALWAYS better.

:)
