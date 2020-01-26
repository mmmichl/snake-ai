import math
import random
from enum import Enum, auto

import pygame

BACKGROUND = 0
HEAD = 1
BODY = 2
FOOD = 3
WALL = 4

SNAKE_COLOR = (0, 220, 0)
SNAKE_HEAD_COLOR = (0, 150, 0)
SNAKE_FOOD_COLOR = (150, 0, 0)

COL_MAP = {
    HEAD: SNAKE_HEAD_COLOR,
    BODY: SNAKE_COLOR,
    FOOD: SNAKE_FOOD_COLOR,
}


class Feedback(Enum):
    ATE_FOOD = auto(),
    HIT_WALL = auto(),
    HIT_TAIL = auto(),
    ELSE = auto()
    WOULD_180 = auto()


class Environment:
    cell_size = 30

    rows = 20
    cols = rows

    # calcuate what we need
    width = cols * cell_size
    height = rows * cell_size

    grid = [[]]
    snake = [(cols // 2, rows // 2), (cols // 2 - 1, rows // 2)]
    food = None

    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")

        self.text = pygame.font.SysFont('Arial', 20)
        self.banner_text = pygame.font.SysFont('Arial', 40)

        self.reset_game()

    def reset_game(self):
        global snake
        snake = [(self.cols // 2, self.rows // 2), (self.cols // 2 - 1, self.rows // 2)]
        self.set_food()
        self.update_grid()

    def get_state(self):
        return self.grid

    def redrawWindow(self):
        self.screen.fill((0, 0, 0))
        # drawGrid(screen)

        self.drawContent(self.grid)
        self.drawScore(self.screen)

        pygame.display.update()

    def step(self, direction: str) -> Feedback:
        x, y = snake[0]

        if direction == 'up':
            y -= 1
        elif direction == 'down':
            y += 1
        elif direction == 'left':
            x -= 1
        elif direction == 'right':
            x += 1

        if self.is_out_of_bound(x, y):
            feedback = Feedback.HIT_WALL
        elif self.bites_in_tail(x, y):
            feedback = Feedback.HIT_TAIL
        else:
            snake.insert(0, (x, y))

            if self.handle_food():
                feedback = Feedback.ATE_FOOD
            else:
                snake.pop()
                feedback = Feedback.ELSE

        self.update_grid()
        return feedback

    ### hepler methods
    def drawBlock(self, x, y, color):
        pygame.draw.rect(self.screen, color,
                         (x * self.cell_size + 1, y * self.cell_size + 1, self.cell_size - 1, self.cell_size - 1))

    def update_grid(self):
        self.grid = [[BACKGROUND for x in range(self.rows)] for y in range(self.cols)]

        for i, s in enumerate(snake):
            self.grid[s[0]][s[1]] = HEAD if i == 0 else BODY

        x, y = self.food
        self.grid[x][y] = FOOD

    def drawContent(self, grid):
        for (x, xar) in enumerate(grid):
            for (y, val) in enumerate(xar):
                if val:
                    self.drawBlock(x, y, COL_MAP[val])

    def drawGrid(self, screen):
        # cellSize = width // rows

        x = 0
        y = 0
        for l in range(max(self.rows, self.cols)):
            x += self.cell_size
            y += self.cell_size

            pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, self.height))
            pygame.draw.line(screen, (255, 255, 255), (0, y), (self.width, y))

    def snake_length(self):
        return len(snake)

    def drawScore(self, screen):
        ts = self.text.render('Score: %d' % len(snake), False, (255, 255, 255))
        screen.blit(ts, (0, 0))

    def bites_in_tail(self, x, y):
        for sx, sy in snake:
            if sx == x and sy == y:
                return True

        return False

    def is_out_of_bound(self, x, y):
        return x < 0 or y < 0 or x >= self.cols or y >= self.cols

    def set_food(self):
        x = min(math.floor(random.random() * self.cols), self.cols - 1)
        y = min(math.floor(random.random() * self.rows), self.rows - 1)

        while self.bites_in_tail(x, y):
            x = min(math.floor(random.random() * self.cols), self.cols - 1)
            y = min(math.floor(random.random() * self.rows), self.rows - 1)

        self.food = (x, y)

    def handle_food(self) -> bool:
        fx, fy = self.food
        sx, sy = snake[0]

        if fx == sx and fy == sy:
            self.set_food()
            return True


def key_to_direction(key):
    if key == pygame.K_LEFT:
        return 'left'
    elif key == pygame.K_RIGHT:
        return 'right'
    elif key == pygame.K_UP:
        return 'up'
    elif key == pygame.K_DOWN:
        return 'down'


def is_valid_direction(current_dir, new_dir):
    if current_dir == 'right' and new_dir == 'left':
        return False
    if current_dir == 'left' and new_dir == 'right':
        return False
    if current_dir == 'down' and new_dir == 'up':
        return False
    if current_dir == 'up' and new_dir == 'down':
        return False

    return True


def handle_keys(key, direction):
    new_dir = key_to_direction(key)

    if not is_valid_direction(direction, new_dir):
        return direction

    return new_dir


class State(Enum):
    INIT = auto(),
    RUN = auto(),
    GAME_OVER = auto(),


def main():
    env = Environment()

    flag = True
    state = State.INIT
    delay = 100
    direction = 'right'

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
            feedback = env.step(direction)
            if feedback in [Feedback.HIT_TAIL, Feedback.HIT_WALL]:
                state = State.GAME_OVER

        env.redrawWindow()
        ts = None
        if state == State.INIT:
            ts = env.banner_text.render('Press \'space\' to start', True, (255, 255, 255), (0, 0, 0))
        if state == State.GAME_OVER:
            ts = env.banner_text.render('Game over!', True, (255, 255, 255), (0, 0, 0))

        if ts:
            w, h = ts.get_size()
            env.screen.blit(ts, (env.width / 2 - w / 2, env.height / 2 - h / 2))

        pygame.display.update()

        # clock.tick(100)
        pygame.time.delay(max(delay, 0))


if __name__ == '__main__':
    main()
