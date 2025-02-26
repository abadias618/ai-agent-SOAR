import json
from read_data import Reader

r = Reader("test")
r.list_games()
game = r.get_one_game('detective')
file = open("timeline_test.txt","a")
for item in game.get_list():
    file.write("state [obs]: ")
    file.write(item["state"]["obs"].strip())
    file.write("\nvalid acts: ")
    file.write(str(item["state"]["valid_acts"].values()))
    file.write("\nscore: ")
    file.write(str(item["state"]["score"]))
    file.write("\nnext state [obs]: ")
    file.write(item["next_state"]["obs"].strip())
    file.write("\nnext state valid acts: ")
    file.write(str(item["next_state"]["valid_acts"].values()))
    file.write("\nnext_state score: ")
    file.write(str(item["next_state"]["score"]))
    file.write("\naction: ")
    file.write(item["action"])
    file.write("\nreward: ")
    file.write(str(item["reward"]))
    file.write("\n\n----------------------------------------------\n\n")
file.close()
    

    
print(json.dumps(game.get_list()[0], indent=3))