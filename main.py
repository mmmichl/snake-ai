import math
import random
from enum import Enum, auto

import pygame

cell_size = 30

rows = 10
cols = rows

SNAKE_COLOR = (0, 220, 0)
SNAKE_HEAD_COLOR = (0, 150, 0)
SNAKE_FOOD_COLOR = (150, 0, 0)

# calcuate what we need
width = cols * cell_size * 2
height = rows * cell_size

snake = [(cols // 2, rows // 2), (cols // 2 - 1, rows // 2)]
food = None

BACKGROUND = 0
HEAD = 1
BODY = 2
FOOD = 3
WALL = 4

COL_MAP = {
    HEAD: SNAKE_HEAD_COLOR,
    BODY: SNAKE_COLOR,
    FOOD: SNAKE_FOOD_COLOR,
}

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake AI")

text = pygame.font.SysFont('Arial', 20)
banner_text = pygame.font.SysFont('Arial', 40)


def drawBlock(x, y, color):
    pygame.draw.rect(screen, color, (x * cell_size + 1, y * cell_size + 1, cell_size - 1, cell_size - 1))


def reset_game():
    global snake
    snake = [(cols // 2, rows // 2), (cols // 2 - 1, rows // 2)]
    set_food()


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


def snake_length():
    return len(snake)


def drawScore(screen):
    ts = text.render('Score: %d' % len(snake), False, (255, 255, 255))
    screen.blit(ts, (0, 0))


def redrawWindow(screen, grid):
    screen.fill((0, 0, 0))
    # drawGrid(screen)

    drawContent(grid)
    drawScore(screen)

    pygame.display.update()


def bites_in_tail(x, y):
    for sx, sy in snake:
        if sx == x and sy == y:
            return True

    return False


def is_out_of_bound(x, y):
    return x < 0 or y < 0 or x >= cols or y >= cols


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


class Feedback(Enum):
    ATE_FOOD = auto(),
    HIT_WALL = auto(),
    HIT_TAIL = auto(),
    ELSE = auto()


def moveSnake(direction: str) -> Feedback:
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
        return Feedback.HIT_WALL

    if bites_in_tail(x, y):
        return Feedback.HIT_TAIL

    snake.insert(0, (x, y))

    if handle_food():
        return Feedback.ATE_FOOD
    else:
        snake.pop()
        return Feedback.ELSE


class State(Enum):
    INIT = auto(),
    RUN = auto(),
    GAME_OVER = auto(),


def handle_keys(key, direction):
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

    return direction


def main():
    set_food()

    clock = pygame.time.Clock()
    flag = True
    state = State.INIT

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
            direction = handle_keys(last_key_event.key, direction)

        if state == State.RUN:
            feedback = moveSnake(direction)
            if feedback in [Feedback.HIT_TAIL, Feedback.HIT_WALL]:
                state = State.GAME_OVER

        grid = setupGrid()
        redrawWindow(screen, grid)
        ts = None
        if state == State.INIT:
            ts = banner_text.render('Press \'space\' to start', True, (255, 255, 255), (0, 0, 0))
        if state == State.GAME_OVER:
            ts = banner_text.render('Game over!', True, (255, 255, 255), (0, 0, 0))

        if ts:
            w, h = ts.get_size()
            screen.blit(ts, (width / 2 - w / 2, height / 2 - h / 2))

        pygame.display.update()

        clock.tick(100)
        pygame.time.delay(max(delay, 0))


if __name__ == '__main__':
    main()
