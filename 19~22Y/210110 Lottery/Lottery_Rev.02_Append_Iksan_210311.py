
import random, os, time
 
def Lotto():
    print("################################\n####### Iksan Lotto #########\n################################\n")
    loop_count = 3
    result = []
 
    for i in range(loop_count):
        temp = []
 
        while True:
            time.sleep(0.01)
 
            if len(temp) == 6:
                break
            else:
                temp2 = random.randint(1,45)
                if temp2 not in temp:
                    temp.append(temp2)
        
        temp.sort()
        
        if temp not in result:
            result.append(temp)
 
    for i in range(len(result)):
        print(i+1,"번째 게임 결과: ", result[i])
 
def Pension():
    print("\n\n################################\n####### Iksan Pension #######\n################################\n")
 
    table = [1,2,3,4,5]
 
    temp_table = random.choice(table)
    print("첫번째 조: ", temp_table)
    table.remove(temp_table)
    print("두번째 조: ", random.choice(table))
 
    print("자동생성 번호:", str(random.randint(0, 999999)).zfill(6))
 
 
if __name__ == "__main__":
    Lotto()
    Pension()
os.system('pause')
