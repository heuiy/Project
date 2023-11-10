#%% lotto
import random
import time
 
def Lotto():
    lotto = [] 
    while len(lotto) != 6:
        num = random.randrange(1, 46)
        if(num in lotto):
            continue
        else:
            lotto.append(num)



    bonus = random.randrange(1, 46)
    while(1):
        if(bonus in lotto):
            bonus = random.randrange(1, 46)
        else:
            break;
            
    lotto.append("+")
    lotto.append(bonus)
    print("로또 예측 번호", end = " = ")
    for i in range(8):
        print(lotto[i], end = ' ')
        time.sleep(0.01)


def Dhlottery():
    dhlottery = []
    first_num = random.randint(1, 5)
    dhlottery.append(first_num)
    dhlottery.append("조")
    while len(dhlottery) != 8:
        num = random.randrange(0, 10)
        dhlottery.append(num)
        
    print("연금복권 예측 번호", end = " = ")
    for i in range(8):
        print(dhlottery[i], end = ' ')
        time.sleep(0.01)
 
 
 
msg = "예측 번호 보기\n① 로또\n② 연금 복권\n③ 나가기\n"
while(1):
    choice = int(input(msg))
    if(choice == 1):
        Lotto()    
        print("\n예측완료\n")
        
    elif(choice == 2):
        Dhlottery()    
        print("\n예측완료\n")
        
    else:
        print("종료")
        break