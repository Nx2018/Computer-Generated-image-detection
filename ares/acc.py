
all_num = 0
ture_num = 0
fal_num = 0
for line in open('./test_result.txt'):
    if line[-2]==line[-4]:
        ture_num += 1
    else :
        fal_num += 1 
    all_num += 1
print all_num
print ture_num
print fal_num