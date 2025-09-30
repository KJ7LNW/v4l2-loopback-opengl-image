import numpy as np


class Camera:
    def __init__(self):
        self.distance = 18.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def get_position(self):
        return np.array([self.pan_x, self.pan_y, self.distance], dtype=np.float32)

    def rotate(self, delta_x, delta_y):
        self.rotation_y += delta_x
        self.rotation_x += delta_y

    def pan(self, delta_x, delta_y):
        self.pan_x += delta_x
        self.pan_y += delta_y

    def zoom(self, delta):
        self.distance = max(1.0, self.distance + delta)

    def get_view_matrix(self):
        view = np.identity(4, dtype=np.float32)
        view = view @ self._translation_matrix(0, 0, -self.distance)
        view = view @ self._translation_matrix(self.pan_x, self.pan_y, 0)
        view = view @ self._rotation_matrix_x(self.rotation_x)
        view = view @ self._rotation_matrix_y(self.rotation_y)
        return view

    def _translation_matrix(self, x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def _rotation_matrix_x(self, angle):
        rad = np.radians(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def _rotation_matrix_y(self, angle):
        rad = np.radians(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)