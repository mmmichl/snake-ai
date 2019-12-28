import math
import random

import numpy as np
import pygame

from main import HEAD, FOOD, set_food, State, handle_keys, moveSnake, redrawWindow, screen, setupGrid, Feedback, \
    snake_length, reset_game, BODY, BACKGROUND, WALL

epsilon = 0.2  # randomization rate
lr = 0.85
gamma = 0.8  # discount factor, typically between 0.8 and 0.99
reward_map = {
    Feedback.HIT_WALL: -1,
    Feedback.HIT_TAIL: -1,
    Feedback.ATE_FOOD: +2,
    Feedback.ELSE: 0,
}

actions = ['up', 'down', 'left', 'right']
action_size = len(actions)


# all 4 direction of the head. with possible values: free, body, border, fruit
S_DIRECTIONS = ['top', 'right', 'bottom', 'left']
S_CELL = ['free', 'body', 'wall', 'food']
cell_map = {
    BACKGROUND: 0,
    BODY: 1,
    WALL: 2,
    FOOD: 3,
}


state_size = len(S_DIRECTIONS) ** len(S_CELL)

q_table = np.zeros((state_size, action_size))


def get_cell(f):
    try:
        return cell_map[f()]
    except:
        return 2

def q_state(grid):
    head_y = head_x = None
    food_y = food_x = None

    for y, y_arr in enumerate(grid):
        for x, s in enumerate(y_arr):
            if s == HEAD:
                head_y, head_x = (y, x)
            elif s == FOOD:
                food_y, food_x = (y, x)

            if head_y and head_x and food_y and food_x:
                break
        if head_y and head_x and food_y and food_x:
            break

    top    = get_cell(lambda: grid[head_y - 1][head_x])
    bottom = get_cell(lambda: grid[head_y + 1][head_x])
    right  = get_cell(lambda: grid[head_y][head_x + 1])
    left   = get_cell(lambda: grid[head_y][head_x - 1])

    return state_to_idx(top, right, bottom, left)


def state_to_idx(top, right, bottom, left):
    return int('%d%d%d%d' % (top, right, bottom, left), 4)


def decide_action(grid) -> int:
    if random.uniform(0, 1) < epsilon:
        # Explore: random action
        action = random.randint(0, action_size - 1)
        # print('random action:', actions[action])
        return action
    else:
        # Exploit: select action with max value
        state = q_table[q_state(grid)]
        action = np.argmax(state)
        # print('action:', actions[action])
        return action


def update_q_values(state, action, new_state, reward):
    q_table[state, action] += lr * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])


def main():
    episode = 1
    max_score = 0

    clock = pygame.time.Clock()
    flag = True
    state = State.RUN
    delay = 200

    reset_game()
    new_grid = setupGrid()

    while flag:
        last_key_event = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.KEYDOWN:
                last_key_event = event
                if event.key == pygame.K_q:
                    pygame.quit()
                    flag = False
                if event.key == pygame.K_EQUALS:
                    delay += 10
                    print('delay', delay)
                if event.key == pygame.K_MINUS:
                    delay -= 10
                    print('delay', delay)
                if event.key == pygame.K_SPACE:
                    state = State.RUN

        # if last_key_event:
        #     direction = handle_keys(last_key_event.key, direction)

        if state == State.GAME_OVER:
            max_score = max(snake_length(), max_score)
            print('Episode %2d, score: %2d/%2d' % (episode, snake_length(), max_score))

            episode += 1
            reset_game()
            new_grid = setupGrid()
            state = State.RUN

        if state == State.RUN:
            grid = new_grid

            action = decide_action(grid)
            feedback = moveSnake(actions[action])
            if feedback in [Feedback.HIT_TAIL, Feedback.HIT_WALL]:
                state = State.GAME_OVER

            reward = reward_map[feedback]
            new_grid = setupGrid()

            update_q_values(q_state(grid), action, q_state(new_grid), reward)

        redrawWindow(screen, new_grid)
        pygame.display.update()

        # clock.tick(100)
        pygame.time.delay(delay)


if __name__ == '__main__':
    main()
