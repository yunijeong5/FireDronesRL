import pygame as pg
import numpy as np
import random

# COPIED FROM REDDIT. IT WORKS--FIND A WAY TO MAP SHAPES/EMOJI, NOT JUST COLOR

# create a 3D array with 30x30x3 (the last dimension is for the RGB color)
cells = np.ndarray((30, 30, 3))

# color dictionary, represents white, red and blue
# TODO: get color RGBs
color_dict = {0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 0, 255)}

# pick a random color tuple from the color dict
for i in range(cells.shape[0]):
    for j in range(cells.shape[1]):
        cells[i][j] = color_dict[random.randrange(3)]

# set the size of the screen as multiples of the array
cellsize = 20
WIDTH = cells.shape[0] * cellsize
HEIGHT = cells.shape[1] * cellsize

# initialize pygame
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()

# create a surface with the size as the array
surf = pg.Surface((cells.shape[0], cells.shape[1]))
# draw the array onto the surface
pg.surfarray.blit_array(surf, cells)
# transform the surface to screen size
surf = pg.transform.scale(surf, (WIDTH, HEIGHT))

# game loop
running = True
while running:
    clock.tick(60)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0))
    # blit the transformed surface onto the screen
    screen.blit(surf, (0, 0))

    pg.display.update()

pg.quit()
