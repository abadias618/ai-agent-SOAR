from jericho import *

# simple walkthrough of the Night game.

env = FrotzEnv("./games/night.z5")
print("\nget_walkthrough()",env.get_walkthrough())
wt = env.get_walkthrough()
for action in wt:
    print("\nstate:\n",env.get_state()[-1].decode("utf-8").strip("\n"))
    print(f"score: {env.get_score()} next action: {action}")
    print(f"valid actions: {env.get_valid_actions()}")
    env.step(action)
    if env.victory() or env.game_over():
        print(f"env.victory() {env.victory()}, env.game_over() {env.game_over()}")
        print("\nFINAL state:\n",env.get_state()[-1].decode("utf-8").strip())
        break
    