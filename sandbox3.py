from uuid import uuid4

def id_generator():
    while True:
        yield str(uuid4())
#g = id_generator()
print(next(id_generator()))
print(id_generator())
print(next(id_generator()))
print(next(id_generator()))
#print(next(g))
#print(next(g))
#print(next(g))

def state_number():
    n = 1
    while True:
        yield n
        n += 1
sn = state_number()    
print(next(sn))
print(next(sn))
print(next(sn))
