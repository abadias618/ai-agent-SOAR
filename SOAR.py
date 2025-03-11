from datetime import datetime
from dotenv import load_dotenv
from jericho import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic_models import *
from State import State
import json

# Extended SOAR architecture
class SOAR(): 
    def __init__(self):
        load_dotenv()
        self.stmem = STMem(self)
        self.ltmem = LTMem(self)
        self.clustering = Clustering(self)
    
    def perception(self, state: State):
        '''
        INPUT: state dict consisting of:
        * ['narrative'] = propmt the game is showing at that state.
        * ['inventory'] = items in inventory (if any).
        * ['location'] = corresponding to the location of the player in the world.
        * ['score'] = current score at that state.
        * ['actions'] = valid actions to take at that state.
        '''
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
        
    def get_curr_state(self) -> State:
        return self.state
    
    def set_curr_state(self, state: State):
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
        
    def router(self, state: State):
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
        # TODO: maybe implement a list to keep track of messages and then
        # add all messages to the json file on __del__()
        dt = datetime.now()
        time = f"{dt.year}-{dt.month}-{dt.day}-{dt.hour}:{dt.minute}:{dt.second}"
        self.filename = str("trial-" + time + ".json")
        with open(self.filename,"w") as f:
            json.dump({"data":[]}, f)
            f.close()
    
    def sem_learning(self, state: State):
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
        response = chain.invoke({"narrative":state.narrative,
                      "inventory":state.inventory,
                      "location":state.location,
                      "actions":state.actions})
        prompt_str = prompt.invoke({"narrative":state.narrative,
                      "inventory":state.inventory,
                      "location":state.location,
                      "actions":state.actions})
        prompt_str = [messgs.content for messgs in prompt_str.messages]
        
        print("prompt",prompt_str)
        print("response",response.response)
        # save response to LT memory
        record = {"prompt":str(" ".join(prompt_str)),
                  "answer":response.response}
        print("record",record)
        with open(self.filename,"r+") as f:
            data = json.load(f)
            data["data"].append(record)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.close()
            
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
