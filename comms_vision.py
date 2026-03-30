import serial
import time
import random
from get_photo import predict_single_frame


# ---------------------------------------------------------
# rps_result(a, b)
#   0 = paper, 1 = rock, 2 = scissors
#   returns +1 if a wins, 0 if tie, -1 if a loses
# ---------------------------------------------------------
def rps_result(a, b):
    if a == b:
        return 0
    return 1 if ((b - a + 3) % 3 == 1) else -1


# ---------------------------------------------------------
# choose_hand(my_right, my_left, opp_left, opp_right)
#
# Returns:
#   0 → remove RIGHT hand (play left)
#   1 → remove LEFT hand  (play right)
#
# This matches your C logic exactly.
# ---------------------------------------------------------
def choose_hand(my_right, my_left, opp_left, opp_right):


    # Payoff matrix
    rr = rps_result(my_right, opp_right)
    rl = rps_result(my_right, opp_left)
    lr = rps_result(my_left,  opp_right)
    ll = rps_result(my_left,  opp_left)

    # Pure strategy dominance
    if rr >= lr and rl >= ll:
        return 1  # always play right
    if lr >= rr and ll >= rl:
        return 0  # always play left

    # Mixed Nash equilibrium
    num   = ll - rl
    denom = (rr - lr) - (rl - ll)
    
    # Ensure denom > 0
    if denom < 0:
        num = -num
        denom = -denom

    # Degenerate case → coin flip
    if denom == 0:
        return random.randint(0, 1)

    # Play right with probability num/denom
    return 1 if random.randrange(denom) < num else 0



#serH = serial.Serial('COM3', 115200, timeout=1)
serB = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # allow ESP32 to reboot

print("Sending test messages...")

test = False
rps = ["rock", "paper", "scissors"]
invRPS = {
        "rock":1,
        "paper":0,
        "scissors":2
    }

while True:
    if(test):
        mse = input("Choose MSE (hand or back): ")
        msg = input("Enter message to send (e.g., left:scissors or right:back): ")
        
        if(mse == "hand"):
            #serH.write((msg + "\n").encode())
            print("Sent:", msg)
        elif (mse == "back"):
            serB.write((msg + "\n").encode())
            print("Sent:", msg)
        
    else:

        # Get randon value
        valueL = random.randint(0, 2)
        valueR = random.randint(0,2)
        print(f"left: {valueL} || right: {valueR}")
        if (valueL == valueR):
            valueR = (valueR + 1) % 3
        #setup hands
        myL = rps[valueL]
        myR = rps[valueR]
        msgL = f"left:{rps[valueL]}"
        msgR = f"right:{rps[valueR]}"
        print(f"left hand go: {msgL} || right hand go: {myR}")
        #serH.write((msgL + "\n").encode())
        time.sleep(2)
        #serH.write((msgR + "\n").encode())

        results = predict_single_frame()
        


        enemyL = results["left"]
        enemyR = results["right"]

        gameLogic = choose_hand(invRPS[myR], invRPS[myL], invRPS[enemyL], invRPS[enemyR])
        if(gameLogic == 0):
            hand = "right"
        else:
            hand = "left"
        print(f"remove {hand} hand")
        msgB = f"{hand}:back"
        serB.write((msgB + "\n").encode())

        wait = input("Enter When Ready...")
        msgB = f"{hand}:forward"
        serB.write((msgB + "\n").encode())


