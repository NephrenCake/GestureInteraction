# -- coding: utf-8 --

def compute_distance(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def compute_direction(x1, y1, x2, y2):
    """
    p2旧 -> p1新
    """
    dx = x1 - x2
    dy = y1 - y2
    if -dx < dy < dx and dx > 0:  # 向右
        return 6
    elif dx < dy < -dx and dx < 0:  # 向左
        return 4
    elif -dy < dx < dy and dy > 0:  # 向下
        return 2
    elif dy < dx < -dy and dy < 0:  # 向上
        return 8
    else:  # 完全没有移动
        return 0
