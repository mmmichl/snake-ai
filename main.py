import math
import random
from enum import Enum, auto

import pygame


cell_size = 20

rows = 20
cols = rows

SNAKE_COLOR = (0, 220, 0)
SNAKE_HEAD_COLOR = (0, 150, 0)
SNAKE_FOOD_COLOR = (150, 0, 0)

# calcuate what we need
width = cols * cell_size
height = rows * cell_size


snake = [(cols // 2, rows // 2)]
direction = 'right'
food = None

BACKGROUND = 0
HEAD = 1
BODY = 2
FOOD = 3

COL_MAP = {
    HEAD: SNAKE_HEAD_COLOR,
    BODY: SNAKE_COLOR,
    FOOD: SNAKE_FOOD_COLOR,
}

pygame.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake AI")


def drawBlock(x, y, color):
    pygame.draw.rect(screen, color, (x * cell_size + 1, y * cell_size + 1, cell_size -1, cell_size -1))


def setupGrid():
    grid = [[BACKGROUND for x in range(rows)] for y in range(cols)]

    for i, s in enumerate(snake):
        grid[s[0]][s[1]] = HEAD if i == 0 else BODY

    x, y = food
    grid[x][y] = FOOD

    return grid


def drawContent(grid):
    for (x, xar) in enumerate(grid):
        for (y, val) in enumerate(xar):
            if val:
                drawBlock(x, y, COL_MAP[val])


def drawGrid(screen):
    # cellSize = width // rows

    x = 0
    y = 0
    for l in range(max(rows, cols)):
        x += cell_size
        y += cell_size

        pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, height))
        pygame.draw.line(screen, (255, 255, 255), (0, y), (width, y))


def redrawWindow(screen):
    screen.fill((0,0,0))
    # drawGrid(screen)

    grid = setupGrid()
    drawContent(grid)

    pygame.display.update()


def bites_in_tail(x, y):
    for sx, sy in snake:
        if sx == x and sy == y:
            return True

    return False


def is_out_of_bound(x, y):
    return x < 0 or y < 0 or x >= cols or y >= cols or bites_in_tail(x, y)


def set_food():
    global food

    x = min(math.floor(random.random() * cols), cols - 1)
    y = min(math.floor(random.random() * rows), rows - 1)

    while bites_in_tail(x, y):
        x = min(math.floor(random.random() * cols), cols - 1)
        y = min(math.floor(random.random() * rows), rows - 1)

    food = (x, y)


def handle_food() -> bool:
    fx, fy = food
    sx, sy = snake[0]

    if fx == sx and fy == sy:
        set_food()
        return True


def moveSnake() -> bool:
    x, y = snake[0]

    if direction == 'up':
        y -= 1
    elif direction == 'down':
        y += 1
    elif direction == 'left':
        x -= 1
    elif direction == 'right':
        x += 1

    if is_out_of_bound(x, y):
        return False

    snake.insert(0, (x,y))

    if not handle_food():
        snake.pop()

    return True


class State(Enum):
    RUN = auto(),
    GAME_OVER = auto(),


def handle_keys(key):
    global direction
    if key == pygame.K_LEFT:
        if not direction == 'right':
            direction = 'left'
    elif key == pygame.K_RIGHT:
        if not direction == 'left':
            direction = 'right'
    elif key == pygame.K_UP:
        if not direction == 'down':
            direction = 'up'
    elif key == pygame.K_DOWN:
        if not direction == 'up':
            direction = 'down'


def main():
    set_food()

    clock = pygame.time.Clock()

    flag = True

    state = State.GAME_OVER

    while flag:
        last_key_event = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.KEYDOWN:
                last_key_event = event
                if event.key == pygame.K_SPACE:
                    state = State.RUN

        if last_key_event:
            handle_keys(last_key_event.key)

        if state == State.RUN:
            if not moveSnake():
                state = State.GAME_OVER

        redrawWindow(screen)

        pygame.time.delay(100)
        clock.tick(10)

        pygame.display.update()


main()
