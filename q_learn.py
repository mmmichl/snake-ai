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
    Feedback.HIT_WALL: -10,
    Feedback.HIT_TAIL: -10,
    Feedback.ATE_FOOD: +10,
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


## for area around head
# state_size = state_base ** (view_size * view_size)
# def q_state(grid, direction) -> int:
#     return q_state_area_head(grid)

state_size = 2 ** 11
def q_state(grid, direction) -> int:
    return q_stead_11_bool(grid, direction)


q_table = np.zeros((state_size, action_size))


def get_cell(f):
    try:
        return cell_map[f()]
    except:
        return 1


def q_stead_11_bool(grid, direction):
    head_x, head_y, food_x, food_y = get_head_position(grid)

    def is_danger(x, y):
        try:
            return grid[x][y] in [BODY, WALL]
        except:
            return True

    dir_up = dir_down = dir_left = dir_right = False

    if direction == 'up':
        danger_ahead = is_danger(head_x - 1, head_y)
        danger_left = is_danger(head_x, head_y - 1)
        danger_right = is_danger(head_x, head_y + 1)
        dir_up = True
    elif direction == 'down':
        danger_ahead = is_danger(head_x + 1, head_y)
        danger_left = is_danger(head_x, head_y + 1)
        danger_right = is_danger(head_x, head_y - 1)
        dir_down = True
    elif direction == 'left':
        danger_ahead = is_danger(head_x, head_y - 1)
        danger_left = is_danger(head_x + 1, head_y)
        danger_right = is_danger(head_x - 1, head_y)
        dir_left = True
    elif direction == 'right':
        danger_ahead = is_danger(head_x, head_y + 1)
        danger_left = is_danger(head_x - 1, head_y)
        danger_right = is_danger(head_x + 1, head_y)
        dir_right = True
    else:
        raise Exception('unknown direction ' + direction)

    food_above = food_x <= head_x
    food_below = food_x >= head_x
    food_left = food_y <= head_y
    food_right = food_y >= head_y

    categorical_state = map(lambda x: int(x), [
        danger_ahead,
        danger_left,
        danger_right,
        dir_up,
        dir_down,
        dir_left,
        dir_right,
        food_above,
        food_below,
        food_left,
        food_right,
    ])

    return int(''.join(map(lambda x: str(x), categorical_state)), 2)


def q_state_area_head(grid) -> int:
    head_x, head_y, _, _ = get_head_position(grid)

    half_view = math.floor(view_size / 2)

    state = []
    for y in range(head_y - half_view, head_y + half_view + 1):
        for x in range(head_x - half_view, head_x + half_view + 1):
            state.append(str(get_cell(lambda: grid[x][y])))

    return int(''.join(state), state_base)


def get_head_position(grid):
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

    return head_x, head_y, food_x, food_y


def state_to_idx(top, right, bottom, left) -> int:
    return int('%d%d%d%d' % (top, right, bottom, left), state_base)


def decide_action(grid, direction) -> int:
    if random.uniform(0, 1) < epsilon:
        # Explore: random action
        action = random.randint(0, action_size - 1)
        # print('random action:', actions[action])
        return action
    else:
        # Exploit: select action with max value
        state = q_table[q_state(grid, direction)]
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

            action = decide_action(grid, direction)
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

            update_q_values(q_state(old_grid, direction), action, q_state(grid, direction), reward)

        redrawWindow(screen, grid)
        pygame.display.update()

        # clock.tick(100)
        pygame.time.delay(delay)


if __name__ == '__main__':
    main()
