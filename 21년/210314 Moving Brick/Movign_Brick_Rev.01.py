# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:31:26 2021

@author: 찬채아빠
"""

#Imports everything that needs to be imported
import pygame
import colors as color
import classes as shape
from random import rand

#Initilizes the pygame and clears the console
pygame.init()
def clear():
  print("\033[H\033[2J", end="")
clear()

#Sets the display with a width of 800 and a height of 600. You can change this later.
#An important note: X starts at 0 in the upper left corner of the screen and gets larger to the right. Y also starts in the upper left corner. However unlike most graphs you might be fimilar with the value of Y becomes larger the more you go down instead of up.
display = pygame.display.set_mode((800, 600))

#Creates variable that will give useful imformation inside the while loops
clickPosition = (0, 0)
mouseDown = False
clicked = False
cycle = 0
keys = pygame.key.get_pressed()
ev = 0
passedKeys = None
quit = False

#Most of theses funstions don't contribute to the actually templeplate and you can delete the ones that say they can be deleted
#Toggles a bool changing True to False and False to True (Can be deleted)
def toggleBool(boolean: bool) -> bool:
  if boolean:
    return False
  else:
    return True

#Will test if your mouse is on a rectangle (Do not delete)
def mouseOn(rect: pygame.Rect):
  if (clickPosition[0] >= rect.x and clickPosition[0] <= rect.x + rect.width) and (clickPosition[1] >= rect.y and clickPosition[1] <= rect.y + rect.height):
    return True
  else:
    return False

#Will return True if the cycle is on a certain cycle. For example if you made the number 100 every 100 cycles it would return True (Do not delete)
def everyCertainCycle(number: int, includeZero: bool = True) -> bool:
  if number == 0:
    return False
  if not(includeZero) and cycle == 0:
    return False
  if (cycle / number) == int(cycle / number):
    return True
  else:
    return False

#Put a list of numbers in the parameter and when the cycle and that number are the same it will return True (Do not delete)
def atCycles(cycles: list):
  for index in cycles:
    if cycle == index:
      return True
  else:
    return False

#Put a list of strings in the parameter and it will print each item in order of the list (Only useful for the example code you can delete)
def multiLinePrint(phrases: list):
  if len(phrases) == 0:
    print("")
  for index in phrases:
    print(index)

#Can test if a certain variable is a tuple and what it's length is then will return True if True (Can be deleted)
def isTuple(variable, length: int) -> bool:
  if length <= 1:
    return False
  try:
    variable[length - 1]
  except:
    return False
  else:
    try:
      variable[length]
    except:
      return True
    else:
      return False

#MUST be put at the end of EVERY while loop in your game (CAN NOT BE DELETED)
def end():
  global clicked
  global cycle
  clicked = False
  cycle += 1
  pygame.display.update()

#MUST be put in the middle of EVERY while loop in your game (CAN NOT BE DELETED)
def events():
  global keys
  global ev
  global clicked
  global mouseDown
  global clickPosition
  global passedKeys
  passedKeys = keys
  keys = pygame.key.get_pressed()
  ev = pygame.event.get()
  for event in ev:
    if event.type == pygame.QUIT:
      pygame.quit()

    if event.type == pygame.MOUSEBUTTONDOWN:
      mouseDown = True
      clicked = False
    elif mouseDown and event.type == pygame.MOUSEBUTTONUP:
      clicked = True
      mouseDown = False
  clickPosition = pygame.mouse.get_pos()

#MUST be put at the start of EVERY while loop for your game. The delay parameter tells how long will what that many milliseconds before continuing the loop. It's recommended you make the delay time a small number otherwise your game will be choppy. The backgroundColor is a RBG color that controls the backgroundColor. Parameters must be filled out. (CAN NOT BE DELETED)
def start(delay: int, backgroundColor):
  pygame.time.delay(delay)
  display.fill(backgroundColor)

#Put a rectangle in the parameter and it will return a tuple with the width and height (Not nessasary but it's nice to have)
def rectSize(rectangle: pygame.Rect):
  return (rectangle.width, rectangle.height)

#Put a rectangle in the parameter and it will return a tuple with the x and y position (Not nessasary but it's nice to have)
def rectPos(rectangle: pygame.Rect):
  return (rectangle.x, rectangle.y)

#Put a rectangle in the parameter and it will return a tuple with the x, y, width, and height (Not nessasary but it's nice to have)
def rectTuple(rectangle: pygame.Rect):
  return (*rectPos(rectangle), *rectSize(rectangle))

def touching(rect1, rect2):
  if rect1.x >= rect2.x and rect1.x + rect1.width <= rect2.x + rect2.width and rect1.y >= rect2.y and rect1.y + rect1.height <= rect2.y + rect2.height:
    return True
  return False

sprites = []
class sprite():
  x = 0
  y = 0
  width = 0
  height = 0
  speed = 0
  deathEffect = 0
  color = (0, 0, 0)

  def __init__(self, x: int, y: int, width: int, height: int, speed: int, color):
    global sprites
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    if speed == 0:
      self.speed = 1
    else:
      self.speed = speed
    self.deathEffect = 0
    self.color = color

    sprites.append(self)
  
  def move(self):
    self.x += self.speed

  def create(self):
    shape.Rectangle(self.x, self.y, self.width, self.height, display, self.color).create()

scoreVar = 0
score = shape.Text(10, 10, 25, "", color.white, color.black, display, bold=10)
missedVar = 0
missed = shape.Text(10, score.x + score.textSize + 10, score.textSize, "", color.white, color.black, display, bold=10)
paused = False
dPaused = shape.Text(750, 10, 50, "", color.white, color.black, display)
#Play symbol: ▶
while True:
  start(0, color.black)
  
  events()
  if keys[pygame.K_ESCAPE]:
    pygame.quit()
  if randint(1, 50) == 1 and not paused:
    direction = (-1 if randint(0, 1) == 0 else 1)
    size = randint(5, 15)
    sprite((0 - size * 2.5 if direction == 1 else 800), randint(0, 600), size * 2.5, size, (randint(1, 2) / 5 if direction == 1 else randint(1, 2) * -1 / 5), color.white if randint(1, 50) != 1 else color.red)
  
  deleteList = []
  if not paused:
    for p, i in enumerate(sprites):
      if i.deathEffect == 0:
        i.move()
      elif i.deathEffect < 500:
        for pos, index in enumerate(sprites):
          if index.deathEffect == 0:
            if touching(index, i):
              scoreVar += 1
              deleteList.append(pos)
        i.width += (i.deathEffect / 500 if i.color == color.white else 1)
        i.x -= (i.deathEffect / 500 if i.color == color.white else 1)/ 2
        i.height += (i.deathEffect / 500 if i.color == color.white else 1)
        i.y -= (i.deathEffect / 500 if i.color == color.white else 1)/ 2
        i.deathEffect += 1
      else:
        scoreVar += 1 if i.color == color.white else 20
        del sprites[p]
      if mouseOn(pygame.Rect(i.x, i.y, i.width, i.height)) and clicked and i.deathEffect == 0:
        i.deathEffect = 1
      if i.speed < 0 and i.x + i.width <= 0:
        missedVar += 1
        del sprites[p]
      elif i.speed > 0 and i.x >= 800:
        missedVar += 1
        del sprites[p]
  deleteList.reverse()
  for i in deleteList:
    del sprites[i]
  for i in sprites:
    if i.color == color.white:
      i.create()
  for i in sprites:
    if i.color == color.red:
      i.create()
  if (mouseOn(dPaused.getRect()) and clicked) or (keys[pygame.K_p] and not passedKeys[pygame.K_p]):
    paused = toggleBool(paused)
  
  dPaused.text = "=" if not paused else ">"
  score.text = "{} points".format(scoreVar)
  missed.text = "{} missed".format(missedVar)
  dPaused.create()
  score.create()
  missed.create()
  end()