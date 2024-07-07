import time

delta = 25

num_vars = 11
spec_vars = num_vars*[0]
# print(spec_vars[0:1])


ctr = 0
        
for spec_vars[0] in range(0, delta + 1):
    # print("upper", delta - sum(spec_vars[0:1]) + 1)
    for spec_vars[1] in range(0, delta - sum(spec_vars[0:1]) + 1):
           for spec_vars[2] in range(0, delta - sum(spec_vars[0:2]) + 1):
                for spec_vars[3] in range(0, delta - sum(spec_vars[0:3]) + 1):
        #             for spec_vars[4] in range(0, delta - sum(spec_vars[0:4]) + 1):
        #                 for spec_vars[5] in range(0, delta - sum(spec_vars[0:5]) + 1):
        #                     for spec_vars[6] in range(0, delta - sum(spec_vars[0:6]) + 1):
        #                         for spec_vars[7] in range(0, delta - sum(spec_vars[0:7]) + 1):
        #                             for spec_vars[8] in range(0, delta - sum(spec_vars[0:8]) + 1):
        #                                 for spec_vars[9] in range(0, delta - sum(spec_vars[0:9]) + 1):
        #                                     for spec_vars[10] in range(0, delta - sum(spec_vars[0:10]) + 1):
                                                # for spec_vars[11] in range(0, delta - sum(spec_vars[0:11]) + 1):
                                                ctr +=1
                                                    
print(ctr)                                                    
# 286,097,760
# 834,451,800
# 600 805 296