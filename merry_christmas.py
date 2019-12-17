import numpy as np
import os
import time
from termcolor import colored

clear = lambda: os.system('clear')

def tree_machine(x):
    if x % 2 ==0:
        print(colored(' '*20+'*','yellow'))
        print(colored(' '*19+'***','yellow'))
        print(colored(' '*20+'*','yellow'))
    else:
        print(colored(' '*19+'/*\\','yellow'))
        print(colored(' '*18+'|***|','yellow'))
        print(colored(' '*19+'\\*/','yellow'))
    for i in range(0,20):
        tree_string = ''
        even = 0
        if (i+2) % 5 == 0:
            if (i+2) % 10 == 0:
                even = 1
            tinsel_color = ['blue','white']
            for j in range(0,i*2):
                tree_string += colored('/',tinsel_color[even])
            print(colored((20-i)*' '+tree_string+(20-i)*' ','green'))
        else:
            tree_string = 'w'
            for j in range(0,i*2):
                q = np.random.randint(0,10)
                bulb_color = ['red','yellow']
                if q<3:
                    tree_string+= colored('w','green')
                elif q<6:
                    tree_string+= colored('o','green')
                elif q<8:
                    bulb_index = np.random.randint(0,2)
                    tree_string+= colored('O',bulb_color[int(bulb_index)])
                else:
                    tree_string+= colored('W','green')
            blank= (20-i)
            string = blank*' '+tree_string+blank*' '
            snowflake_1,snowflake_2 = np.random.randint(0,20),np.random.randint(0,20)
            if snowflake_1 < blank:
                string = string[:snowflake_1]+colored('x','white')+string[1+snowflake_1:]
            if snowflake_2 < blank:
                subs = string[-blank:]
                string = string[:-blank]+subs[:snowflake_2]+colored('x','white')+subs[1+snowflake_2:]
            print(colored(string,'green'))
    print(18*' '+'||||'+18*' ')
    for i in range(0,3):
        if i % 2 ==0:
            print(colored((15+i)*' '+'|'*(5-i)*2+(15+i)*' ','red'))
        else:
            print(colored((15+i)*' '+'|'*(5-i)*2+(15+i)*' ','yellow'))
    print(colored('x'*45,'white'))
    print(colored('x'*11+'   MERRY  CHRISTMAS!   '+'x'*11,'white'))
    print(colored('x'*45,'white'))
    time.sleep(1)
    clear()

x = 1
while x < 100:
    tree_machine(x)
    x+=1