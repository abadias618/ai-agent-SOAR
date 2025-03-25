from jericho import *

# simple walkthrough of the Night game.

env = FrotzEnv("./games/night.z5")
print("words",[w.word for w in env.get_dictionary()],"\n\n\n")
wt = env.get_walkthrough()
for action in wt[0:3]:
    print("\nstate:\n",env.get_state()[-1].decode("utf-8").strip("\n"))
    print(f"score: {env.get_score()} next action: {action}")
    print(f"valid actions: {env.get_valid_actions()}")
    print(env.step(action))
    env.step(action)
    if env.victory() or env.game_over():
        print(f"env.victory() {env.victory()}, env.game_over() {env.game_over()}")
        print("\nFINAL state:\n",env.get_state()[-1].decode("utf-8").strip())
        break