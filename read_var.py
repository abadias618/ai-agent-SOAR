from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate

input_variables=['actions', 'inventory', 'location', 'narrative']
input_types={}
partial_variables={}
messages=[
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            input_types={},
            partial_variables={},
            template='You are acting as an Agent that stores semantic memories (Semantic memory stores factual information, including the definition of terms and concepts).'
            ),
            additional_kwargs={}
        ), 
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['actions', 'inventory', 'location', 'narrative'],
            input_types={}, partial_variables={},
            template='We were given some information about the current state of a narrative game (A game that is played through text prompts), we were given:\
                * The current text being displayed on the screen: {narrative}.\
                * The inventory that our game character currently has: {inventory}.\
                * The coordinates for the location in our game world: {location}.\
                * A list of possible actions to take in the future: {actions}.\
                Your task is to generate a list of 5 facts that you can infer from the current state.\
                These can be things like: "If I have a hammer in my inventory, maybe the game expects me to build something.",\
                or "If the narrative talks about the sea, it might be a Sea related game.". The list shoud strictly contain 5 facts.')
        , additional_kwargs={})
    ]
response= ['The game involves a narrative centered around a computer center, suggesting themes of technology and possibly mystery.',
          'The character is in a precarious situation regarding their job, indicating that the game may involve elements of urgency or consequence.',
          'The mention of a missing laser printer implies that the game may include a quest or task related to finding or retrieving items.',
          "The character's current inventory is empty, suggesting that the game may require them to collect items to progress.",
          'The presence of an interactive fiction game being developed by the character hints that the game may include meta-narrative elements or references to game design.'
          ]