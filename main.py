import json
from jericho import *
from SOAR import SOAR
from State import State

env = FrotzEnv("./games/night.z5")
wt = env.get_walkthrough()


soar = SOAR()
state = State()

state.narrative = env.get_state()[-1].decode("utf-8").strip("\n") # [-1] is because the last element in the state is the narrative
state.inventory = [obj.name for obj in env.get_inventory()]
state.location = env.get_player_location().name
state.score = env.get_score()
state.actions = env.get_valid_actions()

soar.perception(state)
action = soar.action(state)
env.step(action)

state.narrative = env.get_state()[-1].decode("utf-8").strip("\n") # [-1] is because the last element in the state is the narrative
state.inventory = [obj.name for obj in env.get_inventory()]
state.location = env.get_player_location().name
state.score = env.get_score()
state.actions = env.get_valid_actions()

soar.perception(state)