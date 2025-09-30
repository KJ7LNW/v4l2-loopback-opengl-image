#!/usr/bin/env python3

import sys
import subprocess
import threading
import queue
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

from card import Card
from cube import Cube
from light import Light
from camera import Camera


class V4L2GL(Gtk.Window):
    def __init__(self, device=None):
        Gtk.Window.__init__(self, title="V4L2 OpenGL Controller")
        self.set_default_size(800, 600)
        self.set_border_width(10)

        self.image_path = "sample.jpg"
        self.last_directory = os.path.dirname(os.path.abspath(self.image_path)) or os.getcwd()
        self.output_device = device
        self.requested_device = device
        self.ffmpeg_process = None
        self.streaming = False
        self.frame_queue = None
        self.frame_ready_event = None
        self.stream_thread = None
        self.params_lock = threading.Lock()

        self.output_width = 3840
        self.output_height = 2160
        self.output_fps = 24
        self.flip_horizontal = True

        self.cube = None
        self.card = None
        self.light = None
        self.camera = None

        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_rotating = False
        self.mouse_panning = False
        self.mouse_light_rotating = False
        self.mouse_moving_card = False

        self.shader_program = None
        self.stream_fbo = None
        self.stream_texture = None
        self.stream_rbo = None
        self.stream_msaa_fbo = None
        self.stream_msaa_texture = None
        self.stream_msaa_rbo = None
        self.msaa_samples = 8
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
        self.start_btn = Gtk.Button(label="Start Stream")
        self.start_btn.connect("clicked", self.start_stream)
        file_hbox.pack_start(self.start_btn, False, False, 0)
        self.stop_btn = Gtk.Button(label="Stop Stream")
        self.stop_btn.connect("clicked", self.stop_stream)
        self.stop_btn.set_sensitive(False)
        file_hbox.pack_start(self.stop_btn, False, False, 0)
        vbox.pack_start(file_hbox, False, False, 0)

        device_hbox = Gtk.HBox(spacing=6)
        device_hbox.pack_start(Gtk.Label(label="Device:"), False, False, 0)
        self.device_combo = Gtk.ComboBoxText()
        self.populate_video_devices()
        self.device_combo.connect("changed", self.on_device_changed)
        device_hbox.pack_start(self.device_combo, False, False, 0)
        device_hbox.pack_start(Gtk.Label(label="Resolution:"), False, False, 0)
        self.resolution_combo = Gtk.ComboBoxText()
        self.resolution_combo.append_text("720p (1280×720)")
        self.resolution_combo.append_text("1080p (1920×1080)")
        self.resolution_combo.append_text("2K (2560×1440)")
        self.resolution_combo.append_text("4K (3840×2160)")
        self.resolution_combo.append_text("5K (5120×2880)")
        self.resolution_combo.append_text("6K (6144×3456)")
        self.resolution_combo.append_text("8K (7680×4320)")
        self.resolution_combo.set_active(3)
        self.resolution_combo.connect("changed", self.on_resolution_changed)
        device_hbox.pack_start(self.resolution_combo, False, False, 0)
        device_hbox.pack_start(Gtk.Label(label="FPS:"), False, False, 0)
        self.fps_combo = Gtk.ComboBoxText()
        self.fps_combo.append_text("15")
        self.fps_combo.append_text("24")
        self.fps_combo.append_text("30")
        self.fps_combo.append_text("60")
        self.fps_combo.set_active(1)
        self.fps_combo.connect("changed", self.on_fps_changed)
        device_hbox.pack_start(self.fps_combo, False, False, 0)
        vbox.pack_start(device_hbox, False, False, 0)

        cube_size_hbox = Gtk.HBox(spacing=6)
        self.cube_size_label = Gtk.Label(label="Cube Size: 24.0 inches")
        self.cube_size_label.set_halign(Gtk.Align.START)
        cube_size_hbox.pack_start(self.cube_size_label, False, False, 0)
        self.cube_size_slider = Gtk.HScale()
        self.cube_size_slider.set_range(12.0, 48.0)
        self.cube_size_slider.set_value(24.0)
        self.cube_size_slider.set_increments(1.0, 6.0)
        self.cube_size_slider.connect("value-changed", self.update_cube_size)
        cube_size_hbox.pack_start(self.cube_size_slider, True, True, 0)
        self.background_checkbox = Gtk.CheckButton(label="Use Background Image")
        self.background_checkbox.set_active(True)
        self.background_checkbox.connect("toggled", self.on_background_toggled)
        cube_size_hbox.pack_start(self.background_checkbox, False, False, 0)
        self.flip_checkbox = Gtk.CheckButton(label="Flip Horizontal")
        self.flip_checkbox.set_active(True)
        self.flip_checkbox.connect("toggled", self.on_flip_toggled)
        cube_size_hbox.pack_start(self.flip_checkbox, False, False, 0)
        reset_btn = Gtk.Button(label="Reset View")
        reset_btn.connect("clicked", self.reset_view)
        cube_size_hbox.pack_start(reset_btn, False, False, 0)
        vbox.pack_start(cube_size_hbox, False, False, 0)

        self.gl_area = Gtk.GLArea()
        self.gl_area.set_size_request(640, 480)
        self.gl_area.connect("realize", self.on_gl_realize)
        self.gl_area.connect("render", self.on_gl_render)
        self.gl_area.set_has_depth_buffer(True)
        self.gl_area.set_auto_render(True)

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

    def populate_video_devices(self):
        import glob
        devices = []
        for dev_path in sorted(glob.glob("/dev/video*")):
            try:
                dev_num = dev_path.replace("/dev/video", "")
                if not dev_num.isdigit():
                    continue

                name_path = f"/sys/class/video4linux/video{dev_num}/name"
                if os.path.exists(name_path):
                    with open(name_path, 'r') as f:
                        dev_name = f.read().strip()
                    label = f"{dev_path} ({dev_name})"
                else:
                    label = dev_path

                devices.append((dev_path, label))
                self.device_combo.append(dev_path, label)

                if self.requested_device and dev_path == self.requested_device:
                    self.device_combo.set_active_id(dev_path)
            except:
                continue

        if self.device_combo.get_active() == -1 and devices:
            self.device_combo.set_active(0)
            first_dev = self.device_combo.get_active_id()
            if first_dev:
                self.output_device = first_dev

    def on_device_changed(self, widget):
        device_id = widget.get_active_id()
        if device_id:
            self.output_device = device_id

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
                if self.card:
                    self.gl_area.make_current()
                    self.card.set_image(self.image_path, self.load_texture)

            self.gl_area.queue_render()

        dialog.destroy()

    def on_resolution_changed(self, widget):
        text = widget.get_active_text()
        if text:
            resolutions = {
                "720p (1280×720)": (1280, 720),
                "1080p (1920×1080)": (1920, 1080),
                "2K (2560×1440)": (2560, 1440),
                "4K (3840×2160)": (3840, 2160),
                "5K (5120×2880)": (5120, 2880),
                "6K (6144×3456)": (6144, 3456),
                "8K (7680×4320)": (7680, 4320),
            }
            if text in resolutions:
                width, height = resolutions[text]
                self.output_width = width & ~1
                self.output_height = height & ~1
                if self.stream_fbo:
                    self.gl_area.make_current()
                    self.init_stream_fbo()

    def on_fps_changed(self, widget):
        text = widget.get_active_text()
        if text:
            try:
                self.output_fps = int(text)
            except ValueError:
                pass

    def on_mouse_press(self, widget, event):
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        if event.button == 1:
            if event.state & Gdk.ModifierType.SHIFT_MASK:
                self.mouse_moving_card = True
            else:
                self.mouse_rotating = True
        elif event.button == 2:
            self.mouse_panning = True
        elif event.button == 3:
            self.mouse_light_rotating = True

        return True

    def on_mouse_release(self, widget, event):
        if event.button == 1:
            self.mouse_rotating = False
            self.mouse_moving_card = False
        elif event.button == 2:
            self.mouse_panning = False
        elif event.button == 3:
            self.mouse_light_rotating = False

        return True

    def update_cube_size(self, widget):
        with self.params_lock:
            if self.cube:
                self.gl_area.make_current()
                self.cube.size = widget.get_value()
                self.cube.create_vao()
                if self.card:
                    z_pos = self.cube.size / 2.0 + self.card.thickness / 2.0
                    self.card.set_position(self.card.position[0], self.card.position[1], z_pos)
        self.cube_size_label.set_text("Cube Size: %.1f inches" % widget.get_value())
        self.gl_area.queue_render()

    def on_background_toggled(self, widget):
        with self.params_lock:
            if self.cube:
                self.gl_area.make_current()
                if widget.get_active():
                    self.cube.set_image("wood-bg.jpg", self.load_texture)
                else:
                    self.cube.set_image(None, lambda _: self.create_black_texture())
        self.gl_area.queue_render()

    def on_flip_toggled(self, widget):
        self.flip_horizontal = widget.get_active()

    def reset_view(self, widget):
        with self.params_lock:
            if self.camera:
                self.camera.distance = 18.0
                self.camera.rotation_x = 0.0
                self.camera.rotation_y = 0.0
                self.camera.pan_x = 0.0
                self.camera.pan_y = 0.0

            if self.light:
                self.light.rotation_x = 0.0
                self.light.rotation_y = 0.0

            if self.card and self.cube:
                z_pos = self.cube.size / 2.0 + self.card.thickness / 2.0
                self.card.set_position(0.0, 0.0, z_pos)

        self.gl_area.queue_render()

    def on_mouse_motion(self, widget, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y

        with self.params_lock:
            if self.mouse_rotating and self.camera:
                self.camera.rotate(dx * 0.2, dy * 0.2)
                self.gl_area.queue_render()
            elif self.mouse_panning and self.camera:
                self.camera.pan(dx * 0.1, -dy * 0.1)
                self.gl_area.queue_render()
            elif self.mouse_light_rotating and self.light:
                self.light.rotate(dx * 0.1, -dy * 0.1)
                self.gl_area.queue_render()
            elif self.mouse_moving_card and self.card:
                self.card.set_position(
                    self.card.position[0] + dx * 0.05,
                    self.card.position[1] - dy * 0.05,
                    self.card.position[2]
                )
                self.gl_area.queue_render()

        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        return True

    def on_mouse_scroll(self, widget, event):
        with self.params_lock:
            if self.camera:
                if event.direction == Gdk.ScrollDirection.UP:
                    self.camera.zoom(-0.5)
                elif event.direction == Gdk.ScrollDirection.DOWN:
                    self.camera.zoom(0.5)

        self.gl_area.queue_render()
        return True

    def load_texture(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)

            return texture_id
        except:
            return None

    def create_black_texture(self):
        black_data = np.zeros((64, 64, 3), dtype=np.uint8)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, black_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_id

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

        float bayerDither8x8(vec2 coord) {
            int x = int(mod(coord.x, 8.0));
            int y = int(mod(coord.y, 8.0));
            int index = x + y * 8;

            float bayer[64] = float[](
                 0.0/64.0, 48.0/64.0, 12.0/64.0, 60.0/64.0,  3.0/64.0, 51.0/64.0, 15.0/64.0, 63.0/64.0,
                32.0/64.0, 16.0/64.0, 44.0/64.0, 28.0/64.0, 35.0/64.0, 19.0/64.0, 47.0/64.0, 31.0/64.0,
                 8.0/64.0, 56.0/64.0,  4.0/64.0, 52.0/64.0, 11.0/64.0, 59.0/64.0,  7.0/64.0, 55.0/64.0,
                40.0/64.0, 24.0/64.0, 36.0/64.0, 20.0/64.0, 43.0/64.0, 27.0/64.0, 39.0/64.0, 23.0/64.0,
                 2.0/64.0, 50.0/64.0, 14.0/64.0, 62.0/64.0,  1.0/64.0, 49.0/64.0, 13.0/64.0, 61.0/64.0,
                34.0/64.0, 18.0/64.0, 46.0/64.0, 30.0/64.0, 33.0/64.0, 17.0/64.0, 45.0/64.0, 29.0/64.0,
                10.0/64.0, 58.0/64.0,  6.0/64.0, 54.0/64.0,  9.0/64.0, 57.0/64.0,  5.0/64.0, 53.0/64.0,
                42.0/64.0, 26.0/64.0, 38.0/64.0, 22.0/64.0, 41.0/64.0, 25.0/64.0, 37.0/64.0, 21.0/64.0
            );

            return bayer[index];
        }

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
            vec3 litColor = texColor.rgb * lighting;

            float dither = (bayerDither8x8(gl_FragCoord.xy) - 0.5) / 255.0;
            litColor += vec3(dither);

            FragColor = vec4(litColor, texColor.a);
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

    def perspective_matrix(self, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def draw_object(self, obj, projection, view):
        if not obj.texture or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        light_loc = glGetUniformLocation(self.shader_program, "lightPos")
        view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")

        light_pos = self.light.get_position()
        view_pos = self.camera.get_position()

        glUniformMatrix4fv(model_loc, 1, GL_TRUE, obj.get_model_matrix())
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)
        glUniform3f(light_loc, light_pos[0], light_pos[1], light_pos[2])
        glUniform3f(view_pos_loc, view_pos[0], view_pos[1], view_pos[2])

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, obj.texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "textureSampler"), 0)

        glBindVertexArray(obj.vao)
        glDrawElements(GL_TRIANGLES, obj.vertex_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def render_scene(self, projection):
        with self.params_lock:
            view = self.camera.get_view_matrix()

        self.draw_object(self.cube, projection, view)
        self.draw_object(self.card, projection, view)


    def on_gl_realize(self, area):
        area.make_current()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_DITHER)

        try:
            glEnable(GL_SAMPLE_SHADING)
            glMinSampleShading(1.0)
        except:
            pass

        glClearColor(0.1, 0.1, 0.1, 1.0)

        self.shader_program = self.create_shader_program()

        self.cube = Cube("wood-bg.jpg", self.load_texture)
        self.cube.create_vao()

        self.card = Card(self.image_path, self.load_texture)
        z_pos = self.cube.size / 2.0 + self.card.thickness / 2.0
        self.card.set_position(0.0, 0.0, z_pos)
        self.card.create_vao()

        self.camera = Camera()
        self.light = Light(self.cube)

    def perspective_matrix(self, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)


    def on_gl_render(self, area, context):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        width = area.get_allocated_width()
        height = area.get_allocated_height()
        aspect = width / height if height > 0 else 1.0

        projection = self.perspective_matrix(45.0, aspect, 1.0, 500.0)
        self.render_scene(projection)

        glFlush()
        return True

    def init_stream_fbo(self):
        width = self.output_width
        height = self.output_height

        if self.stream_msaa_fbo:
            glDeleteFramebuffers(1, [self.stream_msaa_fbo])
        if self.stream_msaa_texture:
            glDeleteTextures([self.stream_msaa_texture])
        if self.stream_msaa_rbo:
            glDeleteRenderbuffers(1, [self.stream_msaa_rbo])
        if self.stream_fbo:
            glDeleteFramebuffers(1, [self.stream_fbo])
        if self.stream_texture:
            glDeleteTextures([self.stream_texture])
        if self.stream_rbo:
            glDeleteRenderbuffers(1, [self.stream_rbo])

        self.stream_msaa_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.stream_msaa_fbo)

        self.stream_msaa_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, self.stream_msaa_texture)
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, self.msaa_samples, GL_RGB8, width, height, GL_TRUE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, self.stream_msaa_texture, 0)

        self.stream_msaa_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.stream_msaa_rbo)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, self.msaa_samples, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.stream_msaa_rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("ERROR: MSAA Framebuffer is not complete!")

        self.stream_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.stream_fbo)

        self.stream_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.stream_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.stream_texture, 0)

        self.stream_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.stream_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.stream_rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("ERROR: Resolve Framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_to_buffer(self):
        width = self.output_width
        height = self.output_height

        if not self.stream_msaa_fbo:
            self.init_stream_fbo()

        old_viewport = glGetIntegerv(GL_VIEWPORT)

        glBindFramebuffer(GL_FRAMEBUFFER, self.stream_msaa_fbo)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = width / height
        projection = self.perspective_matrix(45.0, aspect, 1.0, 500.0)
        self.render_scene(projection)

        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.stream_msaa_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.stream_fbo)
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST)

        glBindFramebuffer(GL_FRAMEBUFFER, self.stream_fbo)
        glFlush()
        glFinish()

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        img = np.flipud(img)
        if self.flip_horizontal:
            img = np.fliplr(img)
        glPixelStorei(GL_PACK_ALIGNMENT, 4)

        glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return img.tobytes()

    def stream_worker(self):
        frame_time = 1.0 / float(self.output_fps)
        last_frame_time = time.time()

        while self.streaming:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time

                if elapsed >= frame_time:
                    last_frame_time = current_time

                    self.frame_ready_event.clear()
                    GLib.idle_add(self.capture_frame_for_stream)

                    if self.frame_ready_event.wait(timeout=1.0):
                        frame_data = self.frame_queue.get(timeout=0.1)
                        if frame_data and self.ffmpeg_process:
                            self.ffmpeg_process.stdin.write(frame_data)
                            self.ffmpeg_process.stdin.flush()
                else:
                    sleep_time = frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                print(f"Error in stream worker: {e}")
                break

    def capture_frame_for_stream(self):
        if not self.streaming:
            return False

        try:
            self.gl_area.make_current()
            frame_data = self.render_to_buffer()
            if frame_data:
                self.frame_queue.put(frame_data)
            self.frame_ready_event.set()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            self.frame_ready_event.set()

        return False


    def start_stream(self, widget):
        self.image_path = self.file_entry.get_text()
        device_id = self.device_combo.get_active_id()
        if device_id:
            self.output_device = device_id

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.output_width}x{self.output_height}",
            "-r", str(self.output_fps),
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

        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_ready_event = threading.Event()
        self.streaming = True

        self.stream_thread = threading.Thread(target=self.stream_worker, daemon=True)
        self.stream_thread.start()

        self.start_btn.set_sensitive(False)
        self.stop_btn.set_sensitive(True)

    def stop_stream(self, widget=None):
        self.streaming = False

        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
            self.stream_thread = None

        if self.frame_queue:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.frame_queue = None

        if self.frame_ready_event:
            self.frame_ready_event.set()
            self.frame_ready_event = None

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
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else None
    window = V4L2GL(device)
    window.show_all()
    Gtk.main()