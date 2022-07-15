string = "WWWWW  W    @ W          ;WW W        @ W      W WW; W          @         W W;SW S S      @            ;            @   W      W ; SS                      ; S                       ;                         ;                         ;S      S    @            ;            @            ;            @            ;@@@@@    @@@@@@@    @@@@@; S          @            ;S S         @            ;   S        @            ;SS                       ; SS                      ;SS                       ;  S                      ;            @            ;            @            ;            @            ;            @            ;S           @            ;"
height = 25
num_states = 3
width = 12
import random
import os
def makeMap(num_states,  height=height, width = width):
    map = []
    objects = ['W', 'S']
    for row in range(height):
        states = []
        for _ in range(num_states):
            if row%3==0:
                
                states.append(''.join([objects[random.randint(0,1)]+' ' for i in range(int(width/2))]))
            else:
                states.append('            ')
        map.append('@'.join(states))
    return ';'.join(map)

map = makeMap(num_states)
print(map.replace(';','\n'))
file_width = width*num_states
with open(f'ai_economist/foundation/scenarios/simple_wood_and_stone/map_txt/multistatemap_{width*num_states}x{height}.txt', 'w') as f:
    f.write(map)

