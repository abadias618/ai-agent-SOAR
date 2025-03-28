from datetime import datetime
from dotenv import load_dotenv
import json
from jericho import *
# langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# faiss
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
# local imports
from pydantic_models import *
from State import State

# prompts.py
from prompts import NO_STATE, NO_LAST_ACTION
# HF
from huggingface_hub import InferenceClient
from PIL import Image
import io

# Extended SOAR architecture
class SOAR(): 
    def __init__(self):
        load_dotenv()
        self.model = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2)
        self.image_client = InferenceClient(
                    provider="hf-inference",
                    api_key=os.getenv("HF"))
        self.openai_emb = OpenAIEmbeddings()
        self.index = faiss.IndexFlatL2() # Flat index is good for few searches (1000-10000)
        self.faiss = FAISS(
                    embedding_function=self.openai_emb,
                    index=self.index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={})
        self.sem_mem_dir = "./semantic_memory"
        self.proc_mem_dir = "./procedural_memory"
        self.epi_mem_dir = "./episodic_memory"
        self.vis_mem_dir = "./visual_memory"
        self.dt = datetime.now()
        self.stmem = STMem(self, self.dt)
        self.ltmem = LTMem(self, self.dt, self.faiss)
        self.clustering = Clustering(self)
        self.reinforcement_learning = ReinforcementLearning()
        
    def id_generator(self):
        n = 1
        while True:
            yield n
            n += 1
        
    def uuid_generator(self):
        while True:
            yield str(uuid4())
        
    def perception(self, state: State):
        '''
        INPUT: state dict consisting of:
        * ['narrative'] = propmt the game is showing at that state.
        * ['inventory'] = items in inventory (if any).
        * ['location'] = corresponding to the location of the player in the world.
        * ['score'] = current score at that state.
        * ['actions'] = valid actions to take at that state.
        '''
        saved_state = self.stmem.get_curr_state()
        self.stmem.set_prev_state(saved_state)
        self.stmem.set_curr_state(state)
        
        # Start loop by sending to Clustering Class
        prev_state = self.stmem.get_prev_state()
        self.clustering.router(state, prev_state, self.stmem.get_last_action())
        
        # Set the now current as if it were the previous for
        # STMem
        
        return self.stmem.get_curr_state()
    
    def action(self):
        action = self.stmem.decision_proc()
        self.stmem.set_last_action(action)
        return action
    
class STMem():
    '''Short-Term Memory'''
    def __init__(self, soar, dt):
        self.state = None
        self.prev_state = None
        self.last_action = None
        self.soar = soar
        self.app_detect = AppraisalDetector()
        self.mind = {}
        
    def get_curr_state(self) -> State:
        return self.state
    
    def set_curr_state(self, state: State):
        self.state = state
        return self.state
    
    def get_prev_state(self) -> State:
        return self.prev_state
    
    def set_prev_state(self, prev_state: State):
        self.prev_state = prev_state
        return self.prev_state
    
    def get_last_action(self) -> str:
        return self.last_action
    
    def set_last_action(self, action: str) -> str:
        self.last_action = action
        return self.last_action
    
    def edit_mind(self, d):
        self.mind = d
        return
    
    def decision_proc(self):
        return "ne"
    
class AppraisalDetector():
    '''Evaluate Emotion'''
    def __init__(self):
        pass
    
class ReinforcementLearning():
    '''Will consider the scores of previous memories'''
    def __init__(self):
        self.curr = None
        self.prev = None
    
    def set(self, curr, prev):
        if prev == None:
            prev = 0
        self.curr = curr
        self.prev = prev
        
    def get_reward(self):
        res = ""
        if self.prev > self.curr:
            res = f"Your resulting score went down from {self.prev} to {self.curr}, probably a good idea to go back or do something different."
        elif self.prev < self.curr:
            res = f"Your resulting score went up from {self.prev} to {self.curr}, you took a great decision!"
        else:
            res = f"Your resulting score stayed the same, it's not bad, but let's try to take a decision that will get us more points."
        return res
        
class LTMem():
    '''Long-Term Memory'''
    def __init__(self, soar, dt): 
        self.sem_mem = SematicMem(soar.sem_mem_dir, soar.faiss, soar.model, dt, soar.uuid_generator)
        self.proc_mem = ProceduralMem(soar.proc_mem_dir, soar.faiss, soar.model, dt, soar.id_generator, soar.uuid_generator)
        self.epi_mem = EpisodicMem(soar.epi_mem_dir, soar.faiss, soar.model, dt, soar.id_generator, soar.uuid_generator)
        self.lt_vis_mem = LTVisualMem(soar.vis_mem_dir, soar.faiss, soar.image_client, dt)
        self.soar = soar

class Clustering():
    ''' Initially it'll just route the input to the respective
    components, since we're not dealing with statistical data'''
    def __init__(self, soar):
        self.inp = None
        self.soar = soar
        
    def router(self, state: State, prev_state: State = None, last_action: str = None):
        # all of the memories will be tied together based on this state step number
        state_number = next(self.soar.id_gen)
        # pass knowledge to semantic mem (LT mem)
        sem = self.soar.ltmem.sem_mem.sem_learning(state, state_number)
        # pass knowledge to procedural mem (LT mem)
        self.soar.reinforcement_learning.set(state.score, prev_state.score if prev_state != None else None) # error handled
        rl = self.soar.reinforcement_learning.get_reward()
        proc = self.soar.ltmem.proc_mem.proc_learning(state, prev_state, last_action, rl, state_number)
        # pass knowledge to episodic mem (LT mem)
        prompt, uuid = self.soar.ltmem.epi_mem.epi_learning(state, last_action, state_number)
        # pass knowledge to LT visual mem (LT mem)
        image = self.soar.ltmem.lt_vis_mem.save_episode(state_number, prompt)
        self.soar.ltmem.lt_vis_mem.add_image_to_epi_vec(uuid)
        # pass knowledge to ST visual mem
        self.soar.st_vis_mem.save_episode()
        # finally, pass knowledge to ST mem
        d ={"sem":sem, "rl":rl, "proc":proc, "epi":prompt, "image":image}
        self.soar.stmem.edit_mind(d)
class SematicMem():
    def __init__(self, save_dir, faiss, model, datetime_obj, uuid_gen):
        self.model = model
        self.faiss = faiss
        self.uuid_gen = uuid_gen
        self.system_prompt = \
            """You are acting as an Agent that stores semantic memories
            (Semantic memory stores factual information, including the
            definition of terms and concepts)."""
        # TODO: maybe implement a list to keep track of messages and then
        # add all messages to the json file on __del__()
        dt = datetime_obj
        time = f"{dt.year}-{dt.month}-{dt.day}-{dt.hour}:{dt.minute}:{dt.second}"
        self.filename = str(save_dir+"/trial-" + time + ".json")
        with open(self.filename,"w") as f:
            json.dump({"data":[]}, f)
            f.close()
            
    def __del__(self):
        """Save the vec_store to disk just in case"""
        #self.vector_store.dump(self.filename)
        pass
    
    def sem_learning(self, state: State, id: int):
        '''Evaluates input (state) to store knowledge into semantic memory.
        Gets some semantic information by using the state as a prompt in
        combination with an instruction'''
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
        
        # save response to LT memory
        record = {"prompt":str(" ".join(prompt_str)),
                  "answer":response.response}
        
        self.store_response_json(record)
        #self.store_response_vec(record["response"], id)
        return record["answer"]
    
    def store_response_json(self, record):
        """Initially I thought I'd have this as a way of debbuging, but
        maybe I can just asynchronoulsy use a free embed model through 
        the json when necessary"""
        with open(self.filename,"r+") as f:
            data = json.load(f)
            data["data"].append(record)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.close()
        return
    
    def store_response_vec(self, answer_only, id):
        """We don't really need a splitter bc the answers are mostly
        short bc of the prompt"""
        docs = [Document(page_content=ans, metadata={"source":"semantic", "state_num":id}) for ans in answer_only]
        uuids = [next(self.uuid_gen()) for _ in range(len(docs))]
        processed = self.faiss.add_documents(documents = docs, ids=uuids)
        return processed

class ProceduralMem():
    def __init__(self, save_dir, openai_emb, model, datetime_obj):
        self.model = model
        self.vector_store = InMemoryVectorStore(openai_emb)
        self.system_prompt = \
            """You are acting as an Agent that stores procedural memories
            (Procedural memory stores how to do things that later become instictive
            like drive a car or play an instrument). In terms of a computer (You),
            these memories will be things that you learn about the procedure that
            you think are important to remember if someone were to do the same task as you.
            """
        dt = datetime_obj
        time = f"{dt.year}-{dt.month}-{dt.day}-{dt.hour}:{dt.minute}:{dt.second}"
        self.filename = str(save_dir+"/trial-" + time + ".json")
        with open(self.filename,"w") as f:
            json.dump({"data":[]}, f)
            f.close()
            
    def proc_learning(self, state: State, prev_state: State, last_action: str, rl: str, id: int):
        if prev_state == None:
            prev_state = NO_STATE
        if last_action == None:
            last_action = NO_LAST_ACTION   
        prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self.system_prompt),
                        ("human", """We transitioned between 2 states in a narrative game
                                (A game that is played through text prompts) after making a decision.
                                What can you learn from the transition? 
                                For example: 
                                -\"The decision I took transported me to a new place, therefore when I make this decision the consequence is to go to a new place.\"
                                -\"My score in the game increased, therefore taking that decision must be advantageous in some cases.\"
                                -\"Based on the consequences from my action, in what direction is the game trying to take me, and what might be the goal I should pursue?\"
                                The decision you took was: \"{action}\" chosen from {valid_actions}
                                And you went from the screen displaying:
                                {prev_state_narrative}
                                To the screen displaying:
                                {state_narrative}
                                In relation to scores: {rl}
                                You should return exactly 5 answers.
                                """
                        )
                    ]
                
                )
        
        llm = self.model.with_structured_output(GetAnswerAsList)
        chain = prompt | llm # generate response and parse into a list.
        response = chain.invoke({"prev_state_narrative":prev_state.narrative,
                      "state_narrative":state.narrative,
                      "rl":rl,
                      "action":last_action,
                      "valid_actions":prev_state.actions})
        
        prompt_str = prompt.invoke({"prev_state_narrative":prev_state.narrative,
                      "state_narrative":state.narrative,
                      "rl":rl,
                      "action":last_action,
                      "valid_actions":prev_state.actions})
        prompt_str = [messgs.content for messgs in prompt_str.messages]
        
        # save response to LT memory
        record = {"prompt":str(" ".join(prompt_str)),
                  "answer":response.response}
        
        self.store_response_json(record)
        #self.store_response_vec(record["response"], id)
        return record["answer"]
    
    def store_response_json(self, record):
        """Initially I thought I'd have this as a way of debbuging, but
        maybe I can just asynchronoulsy use a free embed model through 
        the json when necessary"""
        with open(self.filename,"r+") as f:
            data = json.load(f)
            data["data"].append(record)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.close()
        return
    
    def store_response_vec(self, answer_only, id):
        """We don't really need a splitter bc the answers are mostly
        short bc of the prompt"""
        docs = [Document(page_content=ans, metadata={"source":"procedural", "state_num":id}) for ans in answer_only]
        uuids = [next(self.uuid_gen()) for _ in range(len(docs))]
        processed = self.faiss.add_documents(documents = docs, ids=uuids)
        return processed
    
    
class EpisodicMem():
    def __init__(self, save_dir, openai_emb, model, datetime_obj):
        self.model = model
        self.vector_store = InMemoryVectorStore(openai_emb)
        self.system_prompt = \
            """You are acting as an Agent that stores episodic memories.
            You need to focus on storing information that can be concatenated
            later with other episodic memories.
            """
        dt = datetime_obj
        time = f"{dt.year}-{dt.month}-{dt.day}-{dt.hour}:{dt.minute}:{dt.second}"
        self.filename = str(save_dir+"/trial-" + time + ".json")
        with open(self.filename,"w") as f:
            json.dump({"data":[]}, f)
            f.close()
            
    def epi_learning(self, state: State, last_action: str, id: int):
        if last_action == None:
            last_action = NO_LAST_ACTION
        prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self.system_prompt),
                        ("human", """Here is some information as a snapshot of what is happening
                                in our narrative game:
                                1. The current narrative: {narrative}
                                2. Your score in the game: {score}
                                3. The actions you can choose from: {actions}
                                4. The action you took previously: {last_action}
                                Summarize what is happening and extract the most meaningful
                                information into 1 sentence that will be stored as an episodic memory.
                                Only return 1 sentence with up to 40 words.
                                """
                        )
                    ]
                
                )
        
        llm = self.model.with_structured_output(GetAnswerAsString)
        chain = prompt | llm # generate response and parse into a list.
        response = chain.invoke({"narrative":state.narrative,
                      "score":state.score,
                      "last_action":last_action,
                      "actions":state.actions})
        
        prompt_str = prompt.invoke({"narrative":state.narrative,
                      "score":state.score,
                      "last_action":last_action,
                      "actions":state.actions})
        prompt_str = [messgs.content for messgs in prompt_str.messages]
        
        # save response to LT memory
        record = {"prompt":str(" ".join(prompt_str)),
                  "answer":response.response}
        
        self.store_response_json(record)
        #id = self.store_response_vec(record["response"], id)
        return (response["answer"], id)
    
    def store_response_json(self, record):
        """Initially I thought I'd have this as a way of debbuging, but
        maybe I can just asynchronoulsy use a free embed model through 
        the json when necessary"""
        with open(self.filename,"r+") as f:
            data = json.load(f)
            data["data"].append(record)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.close()
        return
    
    def store_response_vec(self, answer_only, id):
        """We don't really need a splitter bc the answers are mostly
        short bc of the prompt"""
        docs = Document(page_content=answer_only, metadata={"source":"episodic", "state_num":id})
        uuids = next(self.uuid_gen())
        processed = self.faiss.add_documents(documents = docs, ids=uuids)
        return processed[0] # will return a list with only 1 element
    
class STVisualMem():
    '''Will hold 1 image'''
    def __init__(self, hf_client):
        self.hf_client = hf_client

class LTVisualMem():
    '''Will hold all images generated, linked with the episodic memory'''
    def __init__(self, save_dir, faiss, image_client, datetime_obj):
        self.faiss = faiss
        self.save_dir = save_dir
        self.client = image_client
        self.dt = datetime_obj
        self.image_name = None
        
    def save_episode(self, id, prompt):
        
        image = self.client.text_to_image(
                    prompt,
                    model="nerijs/pixel-art-xl",
                )
        time = f"{self.dt.year}-{self.dt.month}-{self.dt.day}-{self.dt.hour}:{self.dt.minute}:{self.dt.second}"
        name = f"{self.save_dir}-{time}-state-{id}.png"
        image.save(name)
        self.image_name = name
        return image
    
    def add_image_to_epi_vec(self, uuid):
        """We don't really need a splitter bc the answers are mostly
        short bc of the prompt"""
        
        doc = self.faiss.docstore.search(uuid)
        if doc:
            doc.metadata["image"] = self.image_name
        else:
            print(f"no doc found with uuid: {uuid}")
        self.faiss.docstore.update(uuid, doc)
        return 