import numpy as np


class Light:
    def __init__(self, cube):
        self.cube = cube
        self.rotation_x = 0.0
        self.rotation_y = 0.0

    def get_position(self):
        distance = self.cube.size * 2.0
        x = distance * np.sin(np.radians(self.rotation_y)) * np.cos(np.radians(self.rotation_x))
        y = distance * np.sin(np.radians(self.rotation_x))
        z = distance * np.cos(np.radians(self.rotation_y)) * np.cos(np.radians(self.rotation_x))
        return np.array([x, y, z], dtype=np.float32)

    def rotate(self, delta_x, delta_y):
        self.rotation_y += delta_x
        self.rotation_x += delta_y