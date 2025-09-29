import numpy as np
from OpenGL.GL import *
import ctypes


class Cube:
    def __init__(self, image_path, texture_loader):
        self.size = 24.0

        self.image_path = image_path
        self.texture = texture_loader(image_path)
        self.vao = None
        self.vertex_count = 36

    def set_image(self, image_path, texture_loader):
        if self.texture:
            glDeleteTextures([self.texture])
        self.image_path = image_path
        self.texture = texture_loader(image_path)

    def create_vao(self):
        s = self.size / 2.0

        vertices = np.array([
            -s, -s,  s,  0.0, 0.0,  0.0, 0.0, 1.0,
             s, -s,  s,  1.0, 0.0,  0.0, 0.0, 1.0,
             s,  s,  s,  1.0, 1.0,  0.0, 0.0, 1.0,
            -s,  s,  s,  0.0, 1.0,  0.0, 0.0, 1.0,

            -s, -s, -s,  0.0, 0.0,  0.0, 0.0, -1.0,
            -s,  s, -s,  1.0, 0.0,  0.0, 0.0, -1.0,
             s,  s, -s,  1.0, 1.0,  0.0, 0.0, -1.0,
             s, -s, -s,  0.0, 1.0,  0.0, 0.0, -1.0,

            -s,  s, -s,  0.0, 0.0,  0.0, 1.0, 0.0,
            -s,  s,  s,  1.0, 0.0,  0.0, 1.0, 0.0,
             s,  s,  s,  1.0, 1.0,  0.0, 1.0, 0.0,
             s,  s, -s,  0.0, 1.0,  0.0, 1.0, 0.0,

            -s, -s, -s,  0.0, 0.0,  0.0, -1.0, 0.0,
             s, -s, -s,  1.0, 0.0,  0.0, -1.0, 0.0,
             s, -s,  s,  1.0, 1.0,  0.0, -1.0, 0.0,
            -s, -s,  s,  0.0, 1.0,  0.0, -1.0, 0.0,

             s, -s, -s,  0.0, 0.0,  1.0, 0.0, 0.0,
             s,  s, -s,  1.0, 0.0,  1.0, 0.0, 0.0,
             s,  s,  s,  1.0, 1.0,  1.0, 0.0, 0.0,
             s, -s,  s,  0.0, 1.0,  1.0, 0.0, 0.0,

            -s, -s, -s,  0.0, 0.0,  -1.0, 0.0, 0.0,
            -s, -s,  s,  1.0, 0.0,  -1.0, 0.0, 0.0,
            -s,  s,  s,  1.0, 1.0,  -1.0, 0.0, 0.0,
            -s,  s, -s,  0.0, 1.0,  -1.0, 0.0, 0.0,
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20,
        ], dtype=np.uint32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(5 * 4))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

        self.vao = vao

    def get_model_matrix(self):
        return np.identity(4, dtype=np.float32)