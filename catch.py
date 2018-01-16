from collections import deque
import random
import numpy as np

class Catch(object):
    '''Environment object for the game of catch

    screen_width=10: int that sets game screen width

    screen_height=10: int that sets game screen height

    game_over_condition={'points':(-1, 1)}: dict that describes all desired game over conditions
        points: can give single value or tuple of values
            single:
                if < 0, game ends when score is <= given value
                if >= 0, game ends when score is >= given value
            tuple:
                game ends if score <= tuple[0] or score >= tuple[1]
        ball_deletions: int that ends the game when number of balls that
        hit the ground >= given value
        frames: int that ends the game when number of frames >= given value

    fps=30.0: float that sets number of frames/second in the game's simluation of real-time

    global_rate=1: float that sets global rate of change for the entire game

    ball_speed=1: float or list of floats that set possible ball speeds.
    Must follow the form: N, where N is a whole number OR 1/N, where N is a whole number
    If length > 1, ball speeds are chosen randomly as new balls are spawned

    ball_spawn_rate=10: float or list of floats that set rates of new ball spawns.
    Must follow the form: N, where N is a whole number OR 1/N, where N is a whole number
    If length > 1, ball spawn rates are chosen randomly as new balls are spawned.

    output_buffer_size=1: int that sets how many previous game states to output on
    an observation.

    mask: describes whether or not to cover part of the game screen when shown to observer
        bool:
            if False, don't cover any of the game screen
            if True, cover all but the topmost and bottommost rows
        int > 0:
            describes number of rows to cover, starting from second-to-bottommost row

    basket_len=3: int that sets length of the basket at the bottom row; must be > 0

    round_tolerance=0.001: float that determines rounding tolerance for any rates
    that are < 1; namely, global_rate, ball_speed, and ball_spawn_rate.

    index_func=lambda i:i: function that determines index values to pull from the buffer
    '''

    def __init__(self, screen_width=10, screen_height=10,
                 game_over_conditions={'points':(-1, 1)},
                 fps=30.0, global_rate=1, ball_speed=1, ball_spawn_rate=10,
                 output_buffer_size=1, mask=0, basket_len=3,
                 round_tolerance=0.001, index_func= lambda i: i):
        # create the game environment
        self.game_screen = np.zeros(shape=(screen_height, screen_width))
        self.output_queue = deque(np.zeros(
            shape=(output_buffer_size -1, self.game_screen.shape[0],
                   self.game_screen.shape[1])), output_buffer_size)
        self.fps = fps
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ball_speed = ball_speed
        self.d_frame = 1.0/self.fps

        # set environment rates
        # tolerance term for being near a whole number
        self.round_tolerance = round_tolerance
        # force global_rate to be of form: Whole number OR 1/n, where
        # n is a whole number
        if (int(global_rate) != global_rate):
            assert abs(
                np.around(global_rate**-1) - global_rate**-1
                ) <= self.round_tolerance, "ERROR: Fract global rate not in tolerance"
        self.global_rate = global_rate
        # if ball speeds or spawn rates are < 1, need to ensure
        # that they roughly follow the form 1/n, where n is a whole number

        # set possible ball speed ranges; how many pixels will ball fall per frame
        self.ball_speed_range = np.asarray(ball_speed).flatten()

        # check all ball speeds for correctness; speeds are either
        # natural numbers, or follow 1/n, where n is whole number
        for bs in self.ball_speed_range:
            # if spawn rate is less than one, still could be correct
            if 0 < bs < 1:
                # checks if 1/bs is close to 1, given tolerance round_tolerance
                assert abs(np.around(bs**-1) - bs**-1) <= self.round_tolerance, "ERROR: Fract ball speed not in tolerance"
            # if spawn rate is > 1, and fractional
            elif int(bs) != bs:
                raise RuntimeError("ERROR: Invalid fractional ball speed")
            elif (bs <= 0):
                raise RuntimeError("ERROR: Ball speed undefined")

        # set rate of new ball spawn rates; how many frames per ball spawn
        self.ball_spawn_rate = np.asarray(ball_spawn_rate).flatten()

        # check all ball spawn rates for correctness; rates are either
        # natural numbers, or follow 1/n, where n is whole number
        for br in self.ball_spawn_rate:
            # if spawn rate is less than one, still could be correct
            if 0 < br < 1:
                # checks if 1/br is close to 1, given tolerance round_tolerance
                assert abs(np.around(br**-1) - br**-1) <= self.round_tolerance, "ERROR: Fract ball spawn rate not in tolerance"
            # if spawn rate is > 1, and fractional
            elif int(br) != br:
                raise RuntimeError("ERROR: Invalid fractional ball spawn rate")
            elif (br <= 0):
                raise RuntimeError("ERROR: Ball spawn rate undefined")

        # Add basket
        # basket[0] is leftmost column position of basket
        # basket[1] is length of basket
        # basket is always in the bottommost row of the screen
        self.basket_len = basket_len
        self.game_over_conditions = game_over_conditions

        # error check on mask being correct type and value
        assert (isinstance(mask, int) and mask >= 0)
        self.mask = mask

        # set up output queue
        self.index_func = index_func
        self.output_buffer_size = output_buffer_size
        self.queue_size = self.index_func(self.output_buffer_size - 1) + 1

        self.reset()

    def reset(self):
        # clear the screen_width
        self.game_screen = np.zeros(shape=(self.screen_height, self.screen_width))

        # __init__ function counts as drawing the first frame
        self.total_frames = 1
        self.frames_until_next_drop = random.choice(self.ball_spawn_rate)

        # create initial balls
        self.balls = []
        # open slots for new balls
        self.open_ball_slots = list(range(0, self.screen_width))
        # spawn initial balls
        # if more than one ball needs to be spawned initially
        if 0 < self.frames_until_next_drop < 1:
            temp_ball_count = 0
            # spawn 1/n balls for every frame
            while abs(1 - temp_ball_count) > self.round_tolerance:
                self._add_new_ball()
                temp_ball_count += self.frames_until_next_drop
            self.frames_until_next_drop = random.choice(self.ball_spawn_rate)
        # otherwise, ball spawns every N frames
        else:
            self._add_new_ball()
            self.frames_until_next_drop -= 1

        self.basket = [
            np.random.randint(low = 0,
                              high = self.game_screen.shape[1] - (self.basket_len - 1)),
            self.basket_len]

        # set up point totals
        self.points = 0
        self.game_over = False
        self._num_balls_deleted_past_frame = 0
        self._total_balls_deleted = self._num_balls_deleted_past_frame

        # set initial basket and ball positions
        self.game_screen[-1][
                self.basket[0]:(self.basket[0] + self.basket[1])] = 1
        for b in self.balls:
            self.game_screen[b.position()[0]][b.position()[1]] = 1

        self.output_queue = deque(np.zeros(shape=(self.queue_size - 1,
                                                  self.screen_height,
                                                  self.screen_width)),
                                  self.queue_size)
        self.output_queue.appendleft(self.game_screen)


    def _get_new_ball_pos(self):
        '''Gets new ball's column position; makes sure isn't already filled with a ball.'''
        # get new starting row for ball
        new_slot = self.open_ball_slots.pop(random.randint(0, len(self.open_ball_slots)-1))

        # returns starting position for a new ball
        # ball's internal position is a float, seen as int to Catch
        return np.array([0, new_slot], dtype=float)


    def _add_new_ball(self):
        '''Adds a new ball to the game, at a random location on the screen
            Appends onto self.balls, and returns ball object created'''
        b = Ball(
            fall_rate = random.choice(self.ball_speed_range),
            position = self._get_new_ball_pos(), tolerance=self.round_tolerance)

        self.balls.append(b)
        return b


    def _update_game_state(self):
        '''Updates the game state by exactly 1 frame'''

        # create a placeholder for a new gamescreen
        new_screen = np.zeros(shape=self.game_screen.shape)

        # sets the new screen's basket values
        new_screen[-1][self.basket[0]:self.basket[0] + self.basket[1]] = 1

        # tracks balls just spawned
        new_balls = []

        # checks to see if game should spawn a new ball
        # if more than one ball needs to be spawned initially
        if 0 < self.frames_until_next_drop < 1:
            self._ball_add_callback()

            temp_ball_count = 0
            # spawn 1/n balls per frame
            while abs(1 - temp_ball_count) > self.round_tolerance:
                b = self._add_new_ball()
                new_balls.append(b)
                temp_ball_count += self.frames_until_next_drop
            # choose new number of frames until next spawn
            self.frames_until_next_drop = random.choice(self.ball_spawn_rate)
        # if frames_til_drop is 0, time to spawn a ball and choose a new spawn rate
        elif self.frames_until_next_drop == 0:
            self._ball_add_callback()
            b = self._add_new_ball()
            new_balls.append(b)
            self.frames_until_next_drop = random.choice(self.ball_spawn_rate)
        # otherwise, get 1 frame closer to spawning a new ball
        else:
            self.frames_until_next_drop -= 1


        # remove balls from this list after the main loop
        # so you don't mess with the self.balls list being iterated on in the main loop
        removed_balls = []

        # main loop; check all balls and update screen
        for b in self.balls:

            # need to keep track of previous ball's position, to open slots for other new balls
            prev_ball_y = int(b.position()[0])

            # Check if ball has been newly spawned; don't increment if it is
            if not(any(b == x for x in new_balls)):
                b.increment(environ_rate=self.global_rate)
                # checks if ball's previous position was top of grid;
                if prev_ball_y == 0:
                    # if ball moved down, then open that slot for future balls
                    if b.position()[0] > 0:
                        self.open_ball_slots.append(b.position()[1])
                    # if ball did not move down because speed < 1,
                    # force it to move and open slot
                    elif (self.mask > 0) and (b.position()[0] == 0):
                        b._position[0] = 1
                        self.open_ball_slots.append(b.position()[1])

            # if ball's position is at the last row or past the screen
            if b.position()[0] >= new_screen.shape[0] - 1:

                # if ball has fallen into the basket, or intersected
                # with basket when going beyond screen
                if (self.basket[0] <= b.position()[1] <=
                    self.basket[0] + self.basket[1] - 1):
                    self.points += b.value
                    removed_balls.append(b)

                # ball has moved past or touched the ground
                # and not caught by basket
                else:
                    self.points -= b.value
                    removed_balls.append(b)

            # ball is not beyond the screen or at the edge of the screen
            else:
                new_screen[b.position()[0]][b.position()[1]] = 1

        # remove deleted balls from self.balls
        self._num_balls_deleted_past_frame = 0
        for br in removed_balls:
            self.balls.remove(br)
            self._num_balls_deleted_past_frame += 1

        self._total_balls_deleted += self._num_balls_deleted_past_frame

        # set env screen to updated screen
        # black out middle of screen if masking
        if self.mask > 0:
            new_screen [-1 - self.mask:-1][:] = 0
        # set env screen
        self.game_screen = new_screen
        # add to output queue
        self.output_queue.pop()
        self.output_queue.appendleft(self.game_screen)

        # update total number of frames
        self.total_frames += 1


    def act(self, action, ground_truth_return=False):
        '''Allow agent to act on the environment

        action: int that is an element of [-1, 0, 1] that sets basket
        movement -1 is left movement, 0 is staying still, and 1 is
        right movement

        ground_truth_return=False: bool that sets whether to return a
        ground truth value; used when doing fully supervised learning,
        as opposed to reinforcement learning

        Returns:
            If ground_truth_return is True:
                Returns diff in points, diff in frame time,
                    game over status, and ground truth
            If ground_truth_return is False:
                Returns diff in points, diff in frame time, and game over status

        '''

        assert action in list((-1, 0, 1)), "ERROR: Invalid action"
        old_points = self.points

        #update basket according to action
        movement = action * self.global_rate
        movement = int(movement)
        #if basket is trying to move left but cannot
        if self.basket[0] + movement < 0:
            self.basket[0] = 0
        #if basket is trying to move right but cannot
        elif self.basket[0] + self.basket[1] + movement > self.game_screen.shape[1]:
            self.basket[0] = self.game_screen.shape[1] - self.basket[1]
        #otherwise, valid movement
        else:
            if abs(np.around(self.basket[0] + movement) - (self.basket[0] + movement)) <= self.round_tolerance:
                self.basket[0] = np.around(self.basket[0] + movement)
            else:
                self.basket[0] = self.basket[0] + movement

        # Update game state
        self._update_game_state()

        # Check for game over
        if self._check_game_over():
            self.game_over = True

        if ground_truth_return:
            return self.points - old_points, self.d_frame, self.game_over, self._ground_truth()
        else:
            return self.points - old_points, self.d_frame, self.game_over


    def observe(self, flatten=False, expand_dim=False):
        '''Returns current game state according to output queue
        flatten=False: boolean that determines whether to return a flattened output
        expand_dim=False: boolean that expands output dimension by 1
        '''
        if flatten is True:
            out = np.stack(self.output_queue[self.index_func(i)] for i in range(self.output_buffer_size)).flatten()

            if expand_dim:
                return np.expand_dims(out, axis=0)
            else:
                return out
        else:
            out = np.stack(self.output_queue[self.index_func(i)] for i in range(self.output_buffer_size))

            if expand_dim:
                return np.expand_dims(out, axis=1)
            else:
                return out

    def _check_game_over(self):
        '''Return whether there is a game over according to self.game_over_conditions'''
        for cond in self.game_over_conditions:
            if cond == 'points':
                if isinstance(self.game_over_conditions[cond], int):
                    if self.points >= self.game_over_conditions[cond]:
                        return True
                elif isinstance(self.game_over_conditions[cond], tuple):
                    if (
                        self.points <= self.game_over_conditions[cond][0] or
                        self.points >= self.game_over_conditions[cond][1]):
                        return True
            elif cond == 'ball_deletions':
                if self._total_balls_deleted >= self.game_over_conditions[cond]:
                    return True
            elif cond == 'frames':
                if self.total_frames >= self.game_over_conditions[cond]:
                    return True
            else:
                raise RuntimeError("ERROR: Invalid game over condition")

        return False

    def _ball_add_callback(self):
        '''Generic callback for when a ball is added'''
        pass

    def _ground_truth(self):
        '''Returns ground truth to train against'''
        for b in self.balls:
            if (b.position()[0] == self.game_screen.shape[0] - 2) and (
                b.position()[1] in list(x for x in range(self.balls[0] + self.balls[1]))):
                return True

        return False


class Ball(object):
    '''Object that represents a ball inside Catch game

        fall_rate=1.0: float that sets fall rate of ball.
        Must be of form: N, where N is a whole number OR 1/N, where N is a whole number
        Note: This must be assured from outside the Ball object.

        position=np.zeros(2): np.ndarray sets position of the ball relative to the Catch game_screen

        reward=1: real number reward for catching the ball; the penalty for missing the ball is -1*reward

        tolerance=0.001: float that sets rounding tolerance for fractional movements
    '''

    def __init__(self, fall_rate=1.0, position=np.zeros(2), reward=1, tolerance=0.001):
        self.fall_rate = fall_rate
        self._position = position
        self.value = reward
        self.tol = tolerance

    def position(self):
        '''Returns int type of current ball position'''
        return np.array(self._position, dtype=int)

    def increment(self, environ_rate=1.0):
        '''Increments ball's position exactly 1 frame
            environ_rate=1.0: float that gives global rate of chance for environment
        '''
        # increment position
        self._position[0] = self._position[0] + (self.fall_rate * environ_rate)

        # if fall rate is < 1, check for rounding within tol level
        if 0 < self.fall_rate < 1:
            if abs(np.around(self._position[0]) - self._position[0]) <= self.tol:
                self._position = np.around(self._position)

        return self.position()


if __name__ == "__main__":

    c = Catch(screen_width=5, screen_height=5,
              game_over_conditions={'points':(-2, 2), 'frames':5},
              ball_spawn_rate=5)
    for i in range(10):
        reward, dt, game_over = c.act(-1)
        print(reward, game_over)
        print(c.observe())
