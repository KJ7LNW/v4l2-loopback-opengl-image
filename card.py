import numpy as np
from OpenGL.GL import *
from PIL import Image
import ctypes


class Card:
    def __init__(self, image_path, texture_loader):
        self.target_height = 5.0
        self.thickness = 0.01

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.image_path = image_path
        self.texture, self.width, self.height = self._load_image(image_path, texture_loader)
        self.vao = None
        self.vertex_count = 36

    def _load_image(self, image_path, texture_loader):
        img = Image.open(image_path)
        img_width, img_height = img.size

        aspect_ratio = img_width / img_height
        height = self.target_height
        width = height * aspect_ratio

        texture = texture_loader(image_path)
        return texture, width, height

    def set_image(self, image_path, texture_loader):
        if self.texture:
            glDeleteTextures([self.texture])
        self.image_path = image_path
        self.texture, self.width, self.height = self._load_image(image_path, texture_loader)
        if self.vao:
            self.create_vao()

    def set_position(self, x, y, z):
        self.position[0] = x
        self.position[1] = y
        self.position[2] = z

    def create_vao(self):
        w = self.width / 2.0
        h = self.height / 2.0
        t = self.thickness / 2.0

        vertices = np.array([
            -w, -h, t,  0.0, 0.0,  0.0, 0.0, 1.0,
             w, -h, t,  1.0, 0.0,  0.0, 0.0, 1.0,
             w,  h, t,  1.0, 1.0,  0.0, 0.0, 1.0,
            -w,  h, t,  0.0, 1.0,  0.0, 0.0, 1.0,

            -w, -h, -t,  0.0, 0.0,  0.0, 0.0, -1.0,
            -w,  h, -t,  1.0, 0.0,  0.0, 0.0, -1.0,
             w,  h, -t,  1.0, 1.0,  0.0, 0.0, -1.0,
             w, -h, -t,  0.0, 1.0,  0.0, 0.0, -1.0,

            -w,  h, -t,  0.0, 0.0,  0.0, 1.0, 0.0,
            -w,  h,  t,  1.0, 0.0,  0.0, 1.0, 0.0,
             w,  h,  t,  1.0, 1.0,  0.0, 1.0, 0.0,
             w,  h, -t,  0.0, 1.0,  0.0, 1.0, 0.0,

            -w, -h, -t,  0.0, 0.0,  0.0, -1.0, 0.0,
             w, -h, -t,  1.0, 0.0,  0.0, -1.0, 0.0,
             w, -h,  t,  1.0, 1.0,  0.0, -1.0, 0.0,
            -w, -h,  t,  0.0, 1.0,  0.0, -1.0, 0.0,

             w, -h, -t,  0.0, 0.0,  1.0, 0.0, 0.0,
             w,  h, -t,  1.0, 0.0,  1.0, 0.0, 0.0,
             w,  h,  t,  1.0, 1.0,  1.0, 0.0, 0.0,
             w, -h,  t,  0.0, 1.0,  1.0, 0.0, 0.0,

            -w, -h, -t,  0.0, 0.0,  -1.0, 0.0, 0.0,
            -w, -h,  t,  1.0, 0.0,  -1.0, 0.0, 0.0,
            -w,  h,  t,  1.0, 1.0,  -1.0, 0.0, 0.0,
            -w,  h, -t,  0.0, 1.0,  -1.0, 0.0, 0.0,
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
        return np.array([
            [1, 0, 0, self.position[0]],
            [0, 1, 0, self.position[1]],
            [0, 0, 1, self.position[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)