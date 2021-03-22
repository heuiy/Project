# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:17:15 2021

@author: 찬채아빠
"""

import blessings
from getkey import getkey

t = blessings.Terminal()

lvl = 1

level = [
"########",
"#     O#",
"########"
]

def load_level(file):
	global boxes
	boxes = []
	print(t.clear())
	p.x, p.y = 1,1
	global level
	try:
		with open(file,"r") as f:
			level = [i.rstrip("\n") for i in f.readlines()]
	except:
		with open("end.txt","r") as f:
			level = [i.rstrip("\n") for i in f.readlines()]
	for i in range(len(level)):
		for h in range(len(level[i])):
			if level[i][h] == "B":
				level[i] = level[i][:h] + " " + level[i][h+1:]
				box(h,i)

class player:
	def __init__(self):
		self.x = 1
		self.y = 1
	
	def update(self):
		global key
		global level
		hmove = (key == "d")-(key == "a")
		vmove = (key == "s")-(key == "w")
		if level[self.y+vmove][self.x+hmove] != "#":
			pushed = False; col = False
			for i in boxes:
				if i.x == self.x+hmove and i.y == self.y:
					col = True
				if i.x == self.x + hmove and level[self.y][self.x+(hmove*2)] != "#" and self.y == i.y:
					i.x += hmove
					pushed = True
			if (not col) or (col and pushed):
				self.x += hmove
			for i in boxes:
				if i.x == self.x and i.y == self.y+vmove:
					col = True
				if i.y == self.y + vmove and level[self.y+(vmove*2)][self.x] != "#" and self.x == i.x:
					i.y += vmove
					pushed = True
			if (not col) or (col and pushed):
				self.y += vmove

	def draw(self):
		with t.location(self.x,self.y):
			print("§")

boxes = []
class box:
	def __init__(self,x,y):
		self.x = x
		self.y = y
		boxes.append(self)
	
	def draw(self):
		if level[self.y][self.x] == "O":
			boxes.remove(self)
			level[self.y] = level[self.y][:self.x] + " " + level[self.y][self.x+1:]
		with t.location(self.x,self.y):
			print(t.color(30)+"#")

p = player()

key = ""

load_level("level1.txt")

print("\033[?25l")
while True:
	for i in range(len(level)):
		with t.location(0,i):
			print(level[i])
	p.update()
	p.draw()
	for i in boxes:
		i.draw()
	key = getkey()
	if key == "r":
		load_level("level"+str(lvl)+".txt")
	won = True
	for i in level:
		if "O" in i:
			won = False
	if won or key == "o":
		lvl += 1
		load_level("level"+str(lvl)+".txt")