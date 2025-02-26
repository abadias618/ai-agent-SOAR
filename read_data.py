import json

class Reader():
    def __init__(self, split = "train" or "test"):
        self.data = None
        with open(str("./data/"+split+".json"), "r") as file:
            self.data = json.load(file)
    
    def list_games(self):
        l = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j]['rom'] not in l:
                    l.append(self.data[i][j]['rom'])
        print(l)
        print(f"# of games:{len(l)}")
    
    def get_one_game(self, name):
        print(f"Start getting data from {name}...")
        l = []
        switch = False
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j]['rom'] == name:
                    l.append(self.data[i][j])
                    switch = True
            if switch:
                break
        print(f"{len(l)} records returned for game: {name}")
        return OneGameData(l)

class OneGameData():
    def __init__(self, data: list):
        self.data = data
    
    def get_list(self):
        return self.data
        
        