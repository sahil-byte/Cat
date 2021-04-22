# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:54:31 2021

@author: Sahil sutty
"""

nums=[1,7,4,9,2,5]
#p=nums.pop()
#print(p)
cf = 1
last, a = nums[0], None
for i in range(1, len(nums)):
    n = nums[i]
    if n == last: continue
    if a is None:
        cf += 1
        a = n > last
    elif a and n < last or not a and n > last:
        cf += 1
        a = n > last
    last = n
print(cf)
