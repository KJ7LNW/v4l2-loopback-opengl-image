#!/usr/bin/env python3

import sys
import subprocess
import threading
import time
import os
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, Gdk, GLib
from OpenGL.GL import *
from PIL import Image
import numpy as np
import ctypes


class V4L2GL(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="V4L2 OpenGL Controller")
        self.set_default_size(800, 600)
        self.set_border_width(10)

        self.image_path = "sample.jpg"
        self.last_directory = os.path.dirname(os.path.abspath(self.image_path)) or os.getcwd()
        self.output_device = "/dev/video11"
        self.ffmpeg_process = None
        self.stream_thread = None
        self.streaming = False
        self.params_lock = threading.Lock()

        self.output_width = 1920
        self.output_height = 1080

        self.camera_distance = 5.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0

        self.cube_scale = 4.0
        self.flat_scale = 1.0

        self.light_rotation_x = 45.0
        self.light_rotation_y = 45.0

        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_rotating = False
        self.mouse_panning = False
        self.mouse_light_rotating = False

        self.cube_texture = None
        self.flat_texture = None
        self.shader_program = None
        self.cube_vao = None
        self.flat_vao = None
        self.recent_manager = Gtk.RecentManager.get_default()

        self.init_ui()
        self.connect("destroy", self.on_destroy)

    def init_ui(self):
        vbox = Gtk.VBox(spacing=6)
        self.add(vbox)

        file_hbox = Gtk.HBox(spacing=6)
        file_hbox.pack_start(Gtk.Label(label="Image:"), False, False, 0)
        self.file_entry = Gtk.Entry()
        self.file_entry.set_text(self.image_path)
        file_hbox.pack_start(self.file_entry, True, True, 0)
        file_btn = Gtk.Button(label="Browse")
        file_btn.connect("clicked", self.browse_file)
        file_hbox.pack_start(file_btn, False, False, 0)
        vbox.pack_start(file_hbox, False, False, 0)

        device_hbox = Gtk.HBox(spacing=6)
        device_hbox.pack_start(Gtk.Label(label="Device:"), False, False, 0)
        self.device_entry = Gtk.Entry()
        self.device_entry.set_text(self.output_device)
        device_hbox.pack_start(self.device_entry, True, True, 0)
        vbox.pack_start(device_hbox, False, False, 0)

        self.cube_scale_label = Gtk.Label(label="Cube Scale: %.2f" % self.cube_scale)
        self.cube_scale_label.set_alignment(0, 0.5)
        vbox.pack_start(self.cube_scale_label, False, False, 0)
        self.cube_scale_slider = Gtk.HScale()
        self.cube_scale_slider.set_range(0.5, 10.0)
        self.cube_scale_slider.set_value(self.cube_scale)
        self.cube_scale_slider.set_increments(0.1, 1.0)
        self.cube_scale_slider.connect("value-changed", self.update_cube_scale)
        vbox.pack_start(self.cube_scale_slider, False, False, 0)

        self.flat_scale_label = Gtk.Label(label="Image Scale: %.2f" % self.flat_scale)
        self.flat_scale_label.set_alignment(0, 0.5)
        vbox.pack_start(self.flat_scale_label, False, False, 0)
        self.flat_scale_slider = Gtk.HScale()
        self.flat_scale_slider.set_range(0.1, 5.0)
        self.flat_scale_slider.set_value(self.flat_scale)
        self.flat_scale_slider.set_increments(0.05, 0.5)
        self.flat_scale_slider.connect("value-changed", self.update_flat_scale)
        vbox.pack_start(self.flat_scale_slider, False, False, 0)

        self.gl_area = Gtk.GLArea()
        self.gl_area.set_size_request(640, 480)
        self.gl_area.connect("realize", self.on_gl_realize)
        self.gl_area.connect("render", self.on_gl_render)
        self.gl_area.set_has_depth_buffer(True)

        self.gl_area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.SCROLL_MASK
        )
        self.gl_area.connect("button-press-event", self.on_mouse_press)
        self.gl_area.connect("button-release-event", self.on_mouse_release)
        self.gl_area.connect("motion-notify-event", self.on_mouse_motion)
        self.gl_area.connect("scroll-event", self.on_mouse_scroll)

        vbox.pack_start(self.gl_area, True, True, 0)

        button_hbox = Gtk.HBox(spacing=6)

        self.start_btn = Gtk.Button(label="Start Stream")
        self.start_btn.connect("clicked", self.start_stream)
        button_hbox.pack_start(self.start_btn, True, True, 0)

        self.stop_btn = Gtk.Button(label="Stop Stream")
        self.stop_btn.connect("clicked", self.stop_stream)
        self.stop_btn.set_sensitive(False)
        button_hbox.pack_start(self.stop_btn, True, True, 0)

        vbox.pack_start(button_hbox, False, False, 0)

    def browse_file(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Select Image",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        if self.last_directory and os.path.isdir(self.last_directory):
            dialog.set_current_folder(self.last_directory)

        filter_images = Gtk.FileFilter()
        filter_images.set_name("Images")
        filter_images.add_mime_type("image/*")
        dialog.add_filter(filter_images)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            self.file_entry.set_text(filename)
            self.image_path = filename
            self.last_directory = os.path.dirname(filename)

            uri = "file://" + os.path.abspath(filename)
            self.recent_manager.add_item(uri)

            with self.params_lock:
                if self.flat_texture:
                    glDeleteTextures([self.flat_texture])
                self.flat_texture = self.load_texture(self.image_path)

            self.gl_area.queue_render()

        dialog.destroy()

    def on_mouse_press(self, widget, event):
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        if event.button == 1:
            self.mouse_rotating = True
        elif event.button == 2:
            self.mouse_panning = True
        elif event.button == 3:
            self.mouse_light_rotating = True

        return True

    def on_mouse_release(self, widget, event):
        if event.button == 1:
            self.mouse_rotating = False
        elif event.button == 2:
            self.mouse_panning = False
        elif event.button == 3:
            self.mouse_light_rotating = False

        return True

    def update_cube_scale(self, widget):
        with self.params_lock:
            self.cube_scale = widget.get_value()
        self.cube_scale_label.set_text("Cube Scale: %.2f" % self.cube_scale)
        self.gl_area.queue_render()

    def update_flat_scale(self, widget):
        with self.params_lock:
            self.flat_scale = widget.get_value()
        self.flat_scale_label.set_text("Image Scale: %.2f" % self.flat_scale)
        self.gl_area.queue_render()

    def on_mouse_motion(self, widget, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y

        with self.params_lock:
            if self.mouse_rotating:
                self.camera_rotation_y += dx * 0.2
                self.camera_rotation_x += dy * 0.2
                self.gl_area.queue_render()
            elif self.mouse_panning:
                self.camera_pan_x += dx * 0.005
                self.camera_pan_y -= dy * 0.005
                self.gl_area.queue_render()
            elif self.mouse_light_rotating:
                self.light_rotation_y += dx * 0.2
                self.light_rotation_x += dy * 0.2
                self.gl_area.queue_render()

        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        return True

    def on_mouse_scroll(self, widget, event):
        with self.params_lock:
            if event.direction == Gdk.ScrollDirection.UP:
                self.camera_distance = max(2.0, self.camera_distance - 0.2)
            elif event.direction == Gdk.ScrollDirection.DOWN:
                self.camera_distance = min(20.0, self.camera_distance + 0.2)

        self.gl_area.queue_render()
        return True

    def load_texture(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

            return texture_id
        except:
            return None

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compilation failed: {error}")

        return shader

    def create_shader_program(self):
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;
        layout(location = 2) in vec3 normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec2 fragTexCoord;
        out vec3 fragNormal;
        out vec3 fragPos;

        void main()
        {
            gl_Position = projection * view * model * vec4(position, 1.0);
            fragTexCoord = texCoord;
            fragNormal = mat3(model) * normal;
            fragPos = vec3(model * vec4(position, 1.0));
        }
        """

        fragment_shader = """
        #version 330 core
        in vec2 fragTexCoord;
        in vec3 fragNormal;
        in vec3 fragPos;
        out vec4 FragColor;

        uniform sampler2D textureSampler;
        uniform vec3 lightPos;
        uniform vec3 viewPos;

        void main()
        {
            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPos - fragPos);

            float ambient = 0.3;
            float diff = max(dot(norm, lightDir), 0.0);

            vec3 viewDir = normalize(viewPos - fragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * 0.5;

            float lighting = ambient + diff + spec;

            vec4 texColor = texture(textureSampler, fragTexCoord);
            FragColor = vec4(texColor.rgb * lighting, texColor.a);
        }
        """

        vs = self.compile_shader(vertex_shader, GL_VERTEX_SHADER)
        fs = self.compile_shader(fragment_shader, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Program linking failed: {error}")

        glDeleteShader(vs)
        glDeleteShader(fs)

        return program

    def create_cube_vao(self):
        vertices = np.array([
            -1.0, -1.0,  1.0,  0.0, 0.0,  0.0, 0.0, 1.0,
             1.0, -1.0,  1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
             1.0,  1.0,  1.0,  1.0, 1.0,  0.0, 0.0, 1.0,
            -1.0,  1.0,  1.0,  0.0, 1.0,  0.0, 0.0, 1.0,

            -1.0, -1.0, -1.0,  0.0, 0.0,  0.0, 0.0, -1.0,
            -1.0,  1.0, -1.0,  1.0, 0.0,  0.0, 0.0, -1.0,
             1.0,  1.0, -1.0,  1.0, 1.0,  0.0, 0.0, -1.0,
             1.0, -1.0, -1.0,  0.0, 1.0,  0.0, 0.0, -1.0,

            -1.0,  1.0, -1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
            -1.0,  1.0,  1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
             1.0,  1.0,  1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
             1.0,  1.0, -1.0,  0.0, 1.0,  0.0, 1.0, 0.0,

            -1.0, -1.0, -1.0,  0.0, 0.0,  0.0, -1.0, 0.0,
             1.0, -1.0, -1.0,  1.0, 0.0,  0.0, -1.0, 0.0,
             1.0, -1.0,  1.0,  1.0, 1.0,  0.0, -1.0, 0.0,
            -1.0, -1.0,  1.0,  0.0, 1.0,  0.0, -1.0, 0.0,

             1.0, -1.0, -1.0,  0.0, 0.0,  1.0, 0.0, 0.0,
             1.0,  1.0, -1.0,  1.0, 0.0,  1.0, 0.0, 0.0,
             1.0,  1.0,  1.0,  1.0, 1.0,  1.0, 0.0, 0.0,
             1.0, -1.0,  1.0,  0.0, 1.0,  1.0, 0.0, 0.0,

            -1.0, -1.0, -1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,
            -1.0, -1.0,  1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,
            -1.0,  1.0,  1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,
            -1.0,  1.0, -1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,
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

        return vao

    def create_flat_vao(self):
        vertices = np.array([
            -0.8, -0.6, 0.0,  0.0, 0.0,  0.0, 0.0, 1.0,
             0.8, -0.6, 0.0,  1.0, 0.0,  0.0, 0.0, 1.0,
             0.8,  0.6, 0.0,  1.0, 1.0,  0.0, 0.0, 1.0,
            -0.8,  0.6, 0.0,  0.0, 1.0,  0.0, 0.0, 1.0,
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2, 2, 3, 0,
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

        return vao

    def on_gl_realize(self, area):
        area.make_current()

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        self.shader_program = self.create_shader_program()
        self.cube_vao = self.create_cube_vao()
        self.flat_vao = self.create_flat_vao()

        self.cube_texture = self.load_texture("wood-bg.jpg")
        self.flat_texture = self.load_texture(self.image_path)

    def perspective_matrix(self, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def translation_matrix(self, x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def rotation_matrix_x(self, angle):
        rad = np.radians(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def rotation_matrix_y(self, angle):
        rad = np.radians(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def draw_object(self, vao, texture, model_matrix, view_matrix, projection_matrix, light_pos, view_pos):
        if not texture or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        light_loc = glGetUniformLocation(self.shader_program, "lightPos")
        view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")

        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_matrix)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view_matrix)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection_matrix)
        glUniform3f(light_loc, light_pos[0], light_pos[1], light_pos[2])
        glUniform3f(view_pos_loc, view_pos[0], view_pos[1], view_pos[2])

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "textureSampler"), 0)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, 36 if vao == self.cube_vao else 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def scale_matrix(self, sx, sy, sz):
        return np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def on_gl_render(self, area, context):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        width = area.get_allocated_width()
        height = area.get_allocated_height()
        aspect = width / height if height > 0 else 1.0

        with self.params_lock:
            distance = self.camera_distance
            rot_x = self.camera_rotation_x
            rot_y = self.camera_rotation_y
            pan_x = self.camera_pan_x
            pan_y = self.camera_pan_y
            cube_scale = self.cube_scale
            flat_scale = self.flat_scale
            light_rot_x = self.light_rotation_x
            light_rot_y = self.light_rotation_y

        projection = self.perspective_matrix(45.0, aspect, 0.1, 100.0)

        view = np.identity(4, dtype=np.float32)
        view = view @ self.translation_matrix(0, 0, -distance)
        view = view @ self.translation_matrix(pan_x, pan_y, 0)
        view = view @ self.rotation_matrix_x(rot_x)
        view = view @ self.rotation_matrix_y(rot_y)

        light_distance = cube_scale * 2.0
        light_x = light_distance * np.sin(np.radians(light_rot_y)) * np.cos(np.radians(light_rot_x))
        light_y = light_distance * np.sin(np.radians(light_rot_x))
        light_z = light_distance * np.cos(np.radians(light_rot_y)) * np.cos(np.radians(light_rot_x))
        light_pos = [light_x, light_y, light_z]

        view_pos = [pan_x, pan_y, distance]

        cube_model = self.scale_matrix(cube_scale, cube_scale, cube_scale)
        self.draw_object(self.cube_vao, self.cube_texture, cube_model, view, projection, light_pos, view_pos)

        flat_model = self.translation_matrix(0, 0, cube_scale * 1.01)
        flat_model = flat_model @ self.scale_matrix(flat_scale, flat_scale, flat_scale)
        self.draw_object(self.flat_vao, self.flat_texture, flat_model, view, projection, light_pos, view_pos)

        glFlush()
        return True

    def render_to_buffer(self):
        width = self.output_width
        height = self.output_height

        old_viewport = glGetIntegerv(GL_VIEWPORT)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

        rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3])
            return None

        glViewport(0, 0, width, height)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = width / height

        with self.params_lock:
            distance = self.camera_distance
            rot_x = self.camera_rotation_x
            rot_y = self.camera_rotation_y
            pan_x = self.camera_pan_x
            pan_y = self.camera_pan_y
            cube_scale = self.cube_scale
            flat_scale = self.flat_scale
            light_rot_x = self.light_rotation_x
            light_rot_y = self.light_rotation_y

        projection = self.perspective_matrix(45.0, aspect, 0.1, 100.0)

        view = np.identity(4, dtype=np.float32)
        view = view @ self.translation_matrix(0, 0, -distance)
        view = view @ self.translation_matrix(pan_x, pan_y, 0)
        view = view @ self.rotation_matrix_x(rot_x)
        view = view @ self.rotation_matrix_y(rot_y)

        light_distance = cube_scale * 2.0
        light_x = light_distance * np.sin(np.radians(light_rot_y)) * np.cos(np.radians(light_rot_x))
        light_y = light_distance * np.sin(np.radians(light_rot_x))
        light_z = light_distance * np.cos(np.radians(light_rot_y)) * np.cos(np.radians(light_rot_x))
        light_pos = [light_x, light_y, light_z]

        view_pos = [pan_x, pan_y, distance]

        cube_model = self.scale_matrix(cube_scale, cube_scale, cube_scale)
        self.draw_object(self.cube_vao, self.cube_texture, cube_model, view, projection, light_pos, view_pos)

        flat_model = self.translation_matrix(0, 0, cube_scale * 1.01)
        flat_model = flat_model @ self.scale_matrix(flat_scale, flat_scale, flat_scale)
        self.draw_object(self.flat_vao, self.flat_texture, flat_model, view, projection, light_pos, view_pos)

        glFlush()

        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        img = np.flipud(img)

        glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures([texture])
        glDeleteRenderbuffers(1, [rbo])

        return img.tobytes()

    def capture_frame_idle(self):
        if not self.streaming or not self.ffmpeg_process:
            return False

        try:
            self.gl_area.make_current()
            frame_data = self.render_to_buffer()

            if frame_data:
                self.ffmpeg_process.stdin.write(frame_data)
                self.ffmpeg_process.stdin.flush()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            self.stop_stream()
            return False

        return True

    def stream_loop_idle(self):
        return self.capture_frame_idle()

    def start_stream(self, widget):
        self.image_path = self.file_entry.get_text()
        self.output_device = self.device_entry.get_text()

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.output_width}x{self.output_height}",
            "-r", "30",
            "-i", "-",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "32",
            "-f", "v4l2",
            "-vcodec", "rawvideo",
            "-pix_fmt", "yuv420p",
            self.output_device
        ]

        print("Starting FFmpeg with command:", " ".join(cmd))
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=None,
            stderr=None
        )

        self.streaming = True
        GLib.timeout_add(33, self.stream_loop_idle)

        self.start_btn.set_sensitive(False)
        self.stop_btn.set_sensitive(True)

    def stop_stream(self, widget=None):
        self.streaming = False

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                pass
            self.ffmpeg_process = None

        self.start_btn.set_sensitive(True)
        self.stop_btn.set_sensitive(False)

    def on_destroy(self, widget):
        self.stop_stream()
        Gtk.main_quit()


if __name__ == "__main__":
    window = V4L2GL()
    window.show_all()
    Gtk.main()