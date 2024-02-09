# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# 20240209
"""Finger exercise: Write a program that examines three variables—x, y, and z—and
prints the largest odd number among them. If none of them are odd, it should
print a message to that effect."""

# Tests
x, y, z = 0, 0, 0 # 'None of the number is odd'
# x, y, z = 1, 0, 0 # 1
# x, y, z = 0, 1, 0 # 1
# x, y, z = 0, 0, 1 # 1
# x, y, z = 1, 3, 5 # 5
# x, y, z = 3, 3, 1 # 3

result = None

if x%2 == 1:
    
    result = x
    
    if (y%2 == 1) and y > x:
        
        result = y
        
        if (z%2 == 1) and z > y:
            
            result = z
            
    elif  (z%2 == 1) and z > x:
        
        result = z

elif y%2 == 1:
    
    result = y

    if (z%2 == 1) and z > y:
        
        result = z

elif z%2 == 1:
    
    result = z
        
if result:
    
    print(result)
    
else:
    
    print('None of the numbers is odd')

# -

# 20240208
if x%2 == 0:
    if x%3 == 0:
        print('Divisible by 2 and 3')
    else:
        print('Divisible by 2 and not by 3')
elif x%3 == 0:
            print('Divisible by 3 and not by 2')

# 20240208
x = 2
if x%2 == 0:
    print('Even')
else:
    print('Odd')
print('Done with conditional')

# 20240206
# 2 INTRODUCTION TO PYTHON
# 2.1 The Basic Elements of Python
print('Yankees rule!')
print('But not in Boston!')
print('Yankees rule,', 'but not in Boston!')


# 20240205
def Heron(guess, target, tolerance):
    """ Assume guess a strictly positive float
    Assumes target a positive float
    Assumes tolerance a strictly positive float
    prints the results of Heron of Alexandria square root search algorithm"""
    guess = guess
    while abs(guess**2-target)>tolerance:
        guess = (guess+target/guess)/2
        print(abs(guess**2-target))
    print(guess, guess**2, target, abs(guess**2-target))
Heron(0.000001,25,0.0001)
