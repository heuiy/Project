# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:24:33 2021

@author: 찬채아빠
"""

#\033[H\033[2J
from terminalgraphics import *
from readchar import readkey
from random import choice as ch
from os import system as sy

def cls():
    sy('clear')

screen = display(72,25)
back = display(72,25)

legos = [[2,2,'selected'],[10,10,'not_Selected']]

choice = 0
debug = -1
exit_ = -1 
while True:
    screen.fill(' ')
    back.fill(' ')

    for lego in legos:
        lego[2] = 'not_Selected'
        screen.text(lego[0],lego[1],'o-o-o-o\no-o-o-o')
        back.text(lego[0],lego[1],'+ + + +\n+ + + +')

    for lego in legos:
        screen.text(lego[0],lego[1]+2,'|_|_|_|')
        back.text(lego[0],lego[1]+2,'=======')
    if len(legos)>0 and exit_==-1:
        x,y = legos[choice][0],legos[choice][1]
        screen.mark(x-1,y-1,'\\')    
        screen.mark(x+7,y+3,'\\')    
        screen.mark(x-1,y+3,'/')    
        screen.mark(x+7,y-1,'/')    

    screen.show(outline = '#')
    if debug == 1:
        back.show(outline = 'D')
    print('    ^\n |Q|W|E|     |T|\n<|A|S|D|>  (-[%s]+)\n ---V--- |X| |N|'%choice)

    a = readkey()

    cls()
    
    if len(legos)>0:
        legos[choice][2] = 'selected'

    for lego in legos:
        if lego[2] == 'selected':
            if a in ['w','\x1b[A'] and back.check(lego[0]-1,lego[1]+1) != '=' and back.check(lego[0]+6,lego[1]+1) != '=':
                lego[1]-=1
            if a in ['a','\x1b[D'] and back.check(lego[0]-2,lego[1]+2) != '=' and back.check(lego[0],lego[1]+3) != '=' and back.check(lego[0]+6,lego[1]+3) != '=' and back.check(lego[0],lego[1]+4) != '=' and back.check(lego[0]+6,lego[1]+4) != '=' and back.check(lego[0]-1,lego[1]+1) != '=' and back.check(lego[0]+6,lego[1]+1) != '=' and back.check(lego[0]-1,lego[1]) != '=' and back.check(lego[0]+6,lego[1]) != '=':
                lego[0]-=2
            if a in ['s','\x1b[B'] and back.check(lego[0],lego[1]+3) != '=' and back.check(lego[0]+6,lego[1]+3) != '=':
                lego[1]+=1
            if a in ['d','\x1b[C'] and back.check(lego[0]+8,lego[1]+2) != '=' and back.check(lego[0],lego[1]+3) != '=' and back.check(lego[0]+6,lego[1]+3) != '=' and back.check(lego[0],lego[1]+4) != '=' and back.check(lego[0]+6,lego[1]+4) != '=' and back.check(lego[0]-1,lego[1]+1) != '=' and back.check(lego[0]+6,lego[1]+1) != '=' and back.check(lego[0]-1,lego[1]) != '=' and back.check(lego[0]+6,lego[1]) != '=':
                lego[0]+=2
    
    if a == 't':#toggle
        choice+=1
    if a in ['-','_']:
        choice-=1
    if a in ['=','+']:
        choice+=1
    if choice>=len(legos):
        choice=0
    if choice<0:
        choice = len(legos)-1
    if a == 'n':
        legos.append([x+ch([-2,2]),y+ch([-2,2]),''])
        choice = len(legos)-1
    if a == 'e':
        debug*=-1
    if a in ['x','\x7f'] and len(legos)>0:
        a = legos.pop(choice)
        choice-=1
    if a == 'q':
        exit_*=-1