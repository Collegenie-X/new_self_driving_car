### 12.03

> 7_autoplot\_\_\_test_without_servo_motor.py upload

### 7_autoplot\_\_\_test_without_servo_motor.py upload 수정사항

### 라인 코너가 LEFT/RIGHT가 구별되지 않는 경우 (Radom으로, LEFT or RIGHT 선택 )

'''
if (center > up_threshold) :

            random_direction = random.randrange(1,7)
            print ("######### ((( random_direction ))) ################## ")

            if (0< random_direction and random_direction< 4) :  ## 1,2,3일 때 LEFT
                return "LEFT"
            return "RIGHT"

        else :
            return "UP"

'''

### '자율주행*테스트*화면 캡쳐' 폴더 Upload (화면에서 셋팅값을 확인할 수 있습니다.)

'''
print(conner_random_left_2.png)
print(connner_random_left_1.png)
print(left_conner_dectioin_1.png)
print(left_conner_dectioin_2.png)
print(right_conner_dectioin.png)

'''
