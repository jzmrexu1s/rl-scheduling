import math


def timer_fix(original_timer_length, cur_time):
    original_end = math.ceil(cur_time + original_timer_length)
    if original_end % 10 == 1:
        original_end -= 1
    if original_end % 10 == 9:
        original_end += 1
    return original_end - cur_time