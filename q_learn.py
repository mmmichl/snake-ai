import math
import random

import numpy as np
import pygame

from main import HEAD, FOOD, set_food, State, handle_keys, moveSnake, redrawWindow, screen, setupGrid, Feedback, \
    snake_length, reset_game, BODY, BACKGROUND, WALL, rows, cols

epsilon = 0.1  # randomization rate
lr = 0.85
gamma = 0.90  # discount factor, typically between 0.8 and 0.99
reward_map = {
    Feedback.HIT_WALL: -1,
    Feedback.HIT_TAIL: -1,
    Feedback.ATE_FOOD: +1,
    Feedback.ELSE: 0 #-0.1,
}

actions = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, '']
action_size = len(actions)

view_size = 5
assert view_size % 2 == 1

# all 4 direction of the head. with possible values: free, body, border, fruit
S_DIRECTIONS = ['top', 'right', 'bottom', 'left']
S_CELL = ['free', 'body', 'food']
state_base = len(S_CELL)
cell_map = {
    BACKGROUND: 0,
    BODY: 1,
    WALL: 1,
    FOOD: 2,
}


state_size = state_base ** (view_size * view_size)

q_table = np.zeros((state_size, action_size))


def get_cell(f):
    try:
        return cell_map[f()]
    except:
        return 1

def q_state(grid):
    head_y = head_x = None
    food_y = food_x = None

    for x, xarr in enumerate(grid):
        for y, s in enumerate(xarr):
            if s == HEAD:
                head_y, head_x = (y, x)
            elif s == FOOD:
                food_y, food_x = (y, x)

            if head_y and head_x and food_y and food_x:
                break
        if head_y and head_x and food_y and food_x:
            break

    half_view = math.floor(view_size / 2)

    state = []
    for y in range(head_y - half_view, head_y + half_view + 1):
        for x in range(head_x - half_view, head_x + half_view + 1):
            state.append(str(get_cell(lambda: grid[x][y])))

    return int(''.join(state), state_base)


def state_to_idx(top, right, bottom, left):
    return int('%d%d%d%d' % (top, right, bottom, left), state_base)


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


def print_debug(grid, new_grid, action, reward):
    lines_old = []
    lines_new = []
    for y in range(rows):
        l_old = []
        l_new = []
        for x in range(cols):
            l_old.append(str(grid[x][y]))
            l_new.append(str(new_grid[x][y]))

        lines_old.append(l_old)
        lines_new.append(l_new)

    for i in range(len(lines_new)):
        print("%s\t\t%s" % (' '.join(lines_old[i]), ' '.join(lines_new[i])))

    print()
    print('action: "%s", reward %d' % (actions[action], reward))


# def save_model(ep):
#     np.save('snake-%d' % ep, q_table)
#     print('saved')


def main():
    episode = 1
    max_score = 0
    avg_len = np.array([0] * 250)

    clock = pygame.time.Clock()
    flag = True
    state = State.RUN
    delay = 0

    reset_game()
    grid = setupGrid()
    direction = 'right'

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
            death_cuase = 'wall' if feedback == Feedback.HIT_WALL else 'tail' if feedback == Feedback.HIT_TAIL else 'unkn'
            sl = '%2d' % snake_length() if snake_length() > 2 else ' _'
            avg_len[episode % len(avg_len)] = snake_length()
            print('Game %2d, score: %s/%2d, avg len: %.2f' % (episode, sl, max_score, np.average(avg_len)))


            episode += 1
            reset_game()
            grid = setupGrid()
            direction = 'right'
            state = State.RUN

        if state == State.RUN:

            action = decide_action(grid)
            if actions[action]:
                direction = handle_keys(actions[action], direction)
            feedback = moveSnake(direction)

            reward = reward_map[feedback]
            if feedback in [Feedback.HIT_TAIL]:
                # print_debug(old_grid, grid, action, reward)
                state = State.GAME_OVER
            if feedback in [Feedback.HIT_TAIL, Feedback.HIT_WALL]:
                state = State.GAME_OVER
            old_grid = grid
            grid = setupGrid()

            update_q_values(q_state(old_grid), action, q_state(grid), reward)

        redrawWindow(screen, grid)
        pygame.display.update()

        # clock.tick(100)
        pygame.time.delay(delay)


if __name__ == '__main__':
    main()
