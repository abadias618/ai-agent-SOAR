from datetime import datetime
from dotenv import load_dotenv
from jericho import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic_models import *

# Extended SOAR architecture
class SOAR(): 
    def __init__(self, game_dir):
        load_dotenv()
        self.game = FrotzEnv(game_dir)
        self.gold_std = self.game.get_walkthrough()
        self.stmem = STMem(self)
        self.ltmem = LTMem(self)
        self.words = [w.word for w in self.game.get_dictionary()]
        self.clustering = Clustering(self)
    
    def perception(self):
        '''
        INPUT: state dict consisting of:
        * ['narrative'] = propmt the game is showing at that state.
        * ['inventory'] = items in inventory (if any).
        * ['location'] = corresponding to the location of the player in the world.
        * ['score'] = current score at that state.
        * ['actions'] = valid actions to take at that state.
        '''
        state = {}
        
        state['narrative' ] = self.game.get_state()[-1].decode("utf-8").strip("\n") # [-1] is because the last element in the state is the narrative
        state['inventory'] = [obj.name for obj in self.game.get_inventory()]
        state['location'] = self.game.get_player_location().name
        state['score'] = self.game.get_score()
        state['actions'] = self.game.get_valid_actions()
        
        self.stmem.set_curr_state(state)
        
        # Start loop by sending to Clustering Class
        self.clustering.router(state)
        
        return self.stmem.get_curr_state()
    
    def action(self):
        action = self.stmem.decision_proc()
        return action
    
class STMem():
    '''Short-Term Memory'''
    def __init__(self, soar):
        self.state = None
        self.soar = soar
        self.app_detect = AppraisalDetector()
        
    def get_curr_state(self):
        return self.state
    
    def set_curr_state(self, state):
        self.state = state
        return self.state
    
    def decision_proc():
        pass
    
class AppraisalDetector():
    '''Evaluate Emotion'''
    def __init__(self):
        pass
    
class ReinforcementLearning():
    '''Will consider the scores of previous memories'''
    def __init__(self):
        pass
        
class LTMem():
    '''Long-Term Memory'''
    def __init__(self, soar):
        self.sem_mem = SematicMem()
        self.soar = soar

class Clustering():
    ''' Initially it'll just route the input to the respective
    components, since we're not dealing with statistical data'''
    def __init__(self, soar):
        self.inp = None
        self.soar = soar
        
    def router(self, state):
        # pass knowledge to semantic mem (LT mem)
        self.soar.ltmem.sem_mem.sem_learning(state)
        # pass knowledge to episodic mem (LT mem)
        # pass knowledge to procedural mem (LT mem)
        # pass knowledge to LT visual mem (LT mem)
        # pass knowledge to ST visual mem
        # finally, pass knowledge to ST mem

class SematicMem():
    def __init__(self):
        self.model = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2)
        self.system_prompt = \
            """You are acting as an Agent that stores semantic memories
            (Semantic memory stores factual information, including the
            definition of terms and concepts)."""
        self.messages = []
    
    def sem_learning(self, state):
        '''Evaluates input (state) to store knowledge appropiately.'''
        # Build a prompt (piece of text) that stores general knowledge about the state
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", """We were given some information about the current state of a
                            narrative game (A game that is played through text prompts), we
                            were given:
                            * The current text being displayed on the screen: {narrative}.
                            * The inventory that our game character currently has: {inventory}.
                            * The coordinates for the location in our game world: {location}.
                            * A list of possible actions to take in the future: {actions}.
                            
                            Your task is to generate a list of 5 facts that you can infer from the current state.
                            
                            These can be things like: \"If I have a hammer in my inventory, maybe the game expects me to build something.\",
                            or \"If the narrative talks about the sea, it might be a Sea related game.\".
                            
                            The list shoud strictly contain 5 facts."""
                    )
                ]
            
            )
        
        llm = self.model.with_structured_output(GetAnswerAsList)
        chain = prompt | llm # generate response and parse into a list.
        response = chain.invoke({"narrative":state['narrative'],
                      "inventory":state['inventory'],
                      "location":state['location'],
                      "actions":state['actions']})
        print("prompt",prompt)
        print("response",response)
        # save response to LT memory
        with open(str("trial-" + str(datetime.now())),"a") as f:
            f.write(str(prompt))
            f.write(str(response))
        return response
    
    def store_response(self):
        pass
        
        
    
class EpisodicMem():
    def __init__(self):
        pass
    
class STVisualMem():
    def __init__(self):
        pass

class LTVisualMem():
    def __init__(self):
        pass
