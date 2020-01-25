import math
import random

import numpy as np
import pygame

from main import HEAD, FOOD, State, handle_keys, Feedback, BODY, BACKGROUND, WALL, Environment, is_valid_direction

epsilon = 0.2  # randomization rate
annealing_rate = 1.5 # per 50 episodes, devided by
lr = 0.85
gamma = 0.90  # discount factor, typically between 0.8 and 0.99
reward_map = {
    Feedback.HIT_WALL: -10,
    Feedback.HIT_TAIL: -10,
    Feedback.ATE_FOOD: +10,
    Feedback.ELSE: 0,  # -0.1,
    Feedback.WOULD_180: -1,
}

actions = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, '']
action_str = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'nothing']
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
    return q_state_11_bool(grid, direction)


q_table = np.zeros((state_size, action_size))


def get_cell(f):
    try:
        return cell_map[f()]
    except:
        return 1


def q_state_11_bool(grid, direction):
    categorical_state = map(lambda x: int(x), q_state_11(grid, direction))

    return int(''.join(map(lambda x: str(x), categorical_state)), 2)


def q_state_11(grid, direction):
    head_x, head_y, food_x, food_y = get_head_position(grid)

    def is_danger(x, y):
        if x < 0 or y < 0:
            return True
        try:
            return grid[x][y] in [BODY, WALL]
        except:
            return True

    dir_up = dir_down = dir_left = dir_right = False
    if direction == 'up':
        danger_ahead = is_danger(head_x, head_y - 1)
        danger_left = is_danger(head_x + 1, head_y)
        danger_right = is_danger(head_x - 1, head_y)
        dir_up = True
    elif direction == 'down':
        danger_ahead = is_danger(head_x, head_y + 1)
        danger_left = is_danger(head_x - 1, head_y)
        danger_right = is_danger(head_x + 1, head_y)
        dir_down = True
    elif direction == 'left':
        danger_ahead = is_danger(head_x - 1, head_y)
        danger_left = is_danger(head_x, head_y + 1)
        danger_right = is_danger(head_x, head_y - 1)
        dir_left = True
    elif direction == 'right':
        danger_ahead = is_danger(head_x + 1, head_y)
        danger_left = is_danger(head_x, head_y - 1)
        danger_right = is_danger(head_x, head_y + 1)
        dir_right = True
    else:
        raise Exception('unknown direction ' + direction)

    food_above = food_y <= head_y
    food_below = food_y >= head_y
    food_left = food_x <= head_x
    food_right = food_x >= head_x

    return [
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
    ]


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


def decide_action(grid, direction, epsilon) -> (any, int):
    if random.uniform(0, 1) < epsilon:
        # Explore: random action
        action = random.randint(0, action_size - 1)
        # print('random action:', actions[action])
        return 'rnd', action
    else:
        # Exploit: select action with max value
        state = q_table[q_state(grid, direction)]
        action = np.argmax(state)
        # print('action:', actions[action])
        return np.max(state), action


def update_q_values(state, action, new_state, reward):
    q_table[state, action] += lr * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])


def print_debug(grid, new_grid, reason, action, reward, env: Environment, q_state):
    q_state_labels = [
        'danger_ahead',
        'danger_left',
        'danger_right',
        'dir_up',
        'dir_down',
        'dir_left',
        'dir_right',
        'food_above',
        'food_below',
        'food_left',
        'food_right',
    ]
    q_state_output = list(map(lambda x: x[0] + ':\t' + str(x[1]), zip(q_state_labels, q_state)))
    lines_old = []
    lines_new = []
    for y in range(env.rows):
        l_old = []
        l_new = []
        for x in range(env.cols):
            l_old.append(str(grid[x][y]) if grid[x][y] else '·')
            l_new.append(str(new_grid[x][y]) if new_grid[x][y] else '·')

        lines_old.append(l_old)
        lines_new.append(l_new)

    zipped = zip(lines_old, lines_new, q_state_output)
    for o, n, s in zipped:
        print("%s\t\t%s\t\t%s" % (' '.join(o), ' '.join(n), s))

    print('\t' * 12 + q_state_output[10])
    print()
    print('action: "%s" (%s), reward %d' % (action_str[action], reason, reward))


# def save_model(ep):
#     np.save('snake-%d' % ep, q_table)
#     print('saved')


def main():
    episode = 1
    max_score = 0
    avg_len = np.array([0] * 100)
    epsilon_annealed = epsilon

    env = Environment()
    clock = pygame.time.Clock()
    flag = True
    state = State.RUN
    delay = 0
    DEBUG_OUT = False

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
                if event.key == pygame.K_d:
                    DEBUG_OUT = not DEBUG_OUT
                if event.key == pygame.K_SPACE:
                    state = State.RUN

        # if last_key_event:
        #     direction = handle_keys(last_key_event.key, direction)

        if state == State.GAME_OVER:
            max_score = max(env.snake_length(), max_score)
            death_cuase = 'wall' if feedback == Feedback.HIT_WALL else 'tail' if feedback == Feedback.HIT_TAIL else 'unkn'
            sl = '%2d' % env.snake_length() if env.snake_length() > 2 else ' _'
            avg_len[episode % len(avg_len)] = env.snake_length()
            death_by_rnd_action = ', death by random action' if reason == 'rnd' else ''
            print('Game %2d, score: %s/%2d, avg len: %.2f%s' % (episode, sl, max_score, np.average(avg_len), death_by_rnd_action))

            episode += 1
            if episode % 50 == 0 and epsilon_annealed > 0:
                epsilon_annealed /= annealing_rate
                print('new epsilon', epsilon_annealed)

            env.reset_game()
            direction = 'right'
            state = State.RUN

        if state == State.RUN:
            orig_direction = direction
            reason, action = decide_action(env.grid, direction, epsilon_annealed)
            if actions[action]:
                direction = handle_keys(actions[action], direction)
            old_grid = env.grid
            feedback = env.step(direction)

            reward = reward_map[feedback]

            if feedback in [Feedback.HIT_TAIL]:
                state = State.GAME_OVER
            elif feedback in [Feedback.HIT_TAIL, Feedback.HIT_WALL]:
                state = State.GAME_OVER
            elif not is_valid_direction(orig_direction, actions[action]):
                reward = reward_map[Feedback.WOULD_180]

            if DEBUG_OUT:
                print_debug(old_grid, env.grid, reason, action, reward, env, q_state_11(old_grid, orig_direction))
            grid = env.grid

            update_q_values(q_state(old_grid, orig_direction), action, q_state(grid, direction), reward)

        env.redrawWindow()

        # clock.tick(100)
        pygame.time.delay(delay)


if __name__ == '__main__':
    main()
