#!/usr/bin/env python

"""
utils
"""


def get_current_pose(actor):
    """
    Obtains the current x,y, and yaw pose.

    Args:
        actor:

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = actor.get_transform().location.x
    y   = actor.get_transform().location.y
    yaw = math.radians(actor.get_transform().rotation.yaw)

    return (x, y, yaw)
