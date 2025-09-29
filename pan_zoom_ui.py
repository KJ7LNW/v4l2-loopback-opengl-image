#!/usr/bin/env python3

import sys
import subprocess
import threading
import time
import os
import gi
gi.require_version('Gtk', '2.0')
gi.require_version('Gdk', '2.0')
from gi.repository import Gtk, Gdk, GObject
from PIL import Image
import io
import numpy


class PanZoomUI(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="V4L2 Pan/Zoom Controller")
        self.set_default_size(600, 400)
        self.set_border_width(10)

        self.image_path = "sample.jpg"
        self.last_directory = os.path.dirname(os.path.abspath(self.image_path)) or os.getcwd()
        self.output_device = "/dev/video11"
        self.ffmpeg_process = None
        self.stream_thread = None
        self.streaming = False
        self.params_lock = threading.Lock()

        self.default_scale = 800
        self.default_pan_x = 0
        self.default_pan_y = 0
        self.default_phi = 0.0
        self.default_theta = 0.0
        self.default_distance = 2000.0
        self.default_bg_scale = 100.0

        self.scale = self.default_scale
        self.pan_x = self.default_pan_x
        self.pan_y = self.default_pan_y
        self.phi = self.default_phi
        self.theta = self.default_theta
        self.distance = self.default_distance
        self.bg_scale = self.default_bg_scale
        self.output_width = 1920
        self.output_height = 1080

        self.source_image = None
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

        self.scale_label = Gtk.Label(label="Scale: %dpx" % self.scale)
        self.scale_label.set_alignment(0, 0.5)
        vbox.pack_start(self.scale_label, False, False, 0)
        self.scale_slider = Gtk.HScale()
        self.scale_slider.set_range(200, 2400)
        self.scale_slider.set_value(self.scale)
        self.scale_slider.set_increments(10, 100)
        self.scale_slider.connect("value-changed", self.update_scale)
        self.scale_slider.connect("scroll-event", self.on_scroll)
        self.scale_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.scale_slider, False, False, 0)

        self.pan_x_label = Gtk.Label(label="Pan X: %dpx" % self.pan_x)
        self.pan_x_label.set_alignment(0, 0.5)
        vbox.pack_start(self.pan_x_label, False, False, 0)
        self.pan_x_slider = Gtk.HScale()
        self.pan_x_slider.set_range(-960, 960)
        self.pan_x_slider.set_value(self.pan_x)
        self.pan_x_slider.set_increments(5, 50)
        self.pan_x_slider.connect("value-changed", self.update_pan_x)
        self.pan_x_slider.connect("scroll-event", self.on_scroll)
        self.pan_x_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.pan_x_slider, False, False, 0)

        self.pan_y_label = Gtk.Label(label="Pan Y: %dpx" % self.pan_y)
        self.pan_y_label.set_alignment(0, 0.5)
        vbox.pack_start(self.pan_y_label, False, False, 0)
        self.pan_y_slider = Gtk.HScale()
        self.pan_y_slider.set_range(-540, 540)
        self.pan_y_slider.set_value(self.pan_y)
        self.pan_y_slider.set_increments(5, 50)
        self.pan_y_slider.connect("value-changed", self.update_pan_y)
        self.pan_y_slider.connect("scroll-event", self.on_scroll)
        self.pan_y_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.pan_y_slider, False, False, 0)

        self.phi_label = Gtk.Label(label="Phi (vertical tilt): %.2f°" % self.phi)
        self.phi_label.set_alignment(0, 0.5)
        vbox.pack_start(self.phi_label, False, False, 0)
        self.phi_slider = Gtk.HScale()
        self.phi_slider.set_range(-45.0, 45.0)
        self.phi_slider.set_value(self.phi)
        self.phi_slider.set_increments(1.0, 5.0)
        self.phi_slider.connect("value-changed", self.update_phi)
        self.phi_slider.connect("scroll-event", self.on_scroll)
        self.phi_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.phi_slider, False, False, 0)

        self.theta_label = Gtk.Label(label="Theta (horizontal tilt): %.2f°" % self.theta)
        self.theta_label.set_alignment(0, 0.5)
        vbox.pack_start(self.theta_label, False, False, 0)
        self.theta_slider = Gtk.HScale()
        self.theta_slider.set_range(-45.0, 45.0)
        self.theta_slider.set_value(self.theta)
        self.theta_slider.set_increments(1.0, 5.0)
        self.theta_slider.connect("value-changed", self.update_theta)
        self.theta_slider.connect("scroll-event", self.on_scroll)
        self.theta_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.theta_slider, False, False, 0)

        self.distance_label = Gtk.Label(label="Distance: %.0f" % self.distance)
        self.distance_label.set_alignment(0, 0.5)
        vbox.pack_start(self.distance_label, False, False, 0)
        self.distance_slider = Gtk.HScale()
        self.distance_slider.set_range(500.0, 5000.0)
        self.distance_slider.set_value(self.distance)
        self.distance_slider.set_increments(50.0, 500.0)
        self.distance_slider.connect("value-changed", self.update_distance)
        self.distance_slider.connect("scroll-event", self.on_scroll)
        self.distance_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.distance_slider, False, False, 0)

        self.bg_scale_label = Gtk.Label(label="Background Scale: %.0f%%" % self.bg_scale)
        self.bg_scale_label.set_alignment(0, 0.5)
        vbox.pack_start(self.bg_scale_label, False, False, 0)
        self.bg_scale_slider = Gtk.HScale()
        self.bg_scale_slider.set_range(50.0, 200.0)
        self.bg_scale_slider.set_value(self.bg_scale)
        self.bg_scale_slider.set_increments(1.0, 10.0)
        self.bg_scale_slider.connect("value-changed", self.update_bg_scale)
        self.bg_scale_slider.connect("scroll-event", self.on_scroll)
        self.bg_scale_slider.add_events(Gdk.EventMask.SCROLL_MASK)
        vbox.pack_start(self.bg_scale_slider, False, False, 0)

        self.filter_label = Gtk.Label()
        self.filter_label.set_line_wrap(True)
        self.filter_label.set_alignment(0, 0.5)
        vbox.pack_start(self.filter_label, False, False, 0)

        button_hbox = Gtk.HBox(spacing=6)

        reset_btn = Gtk.Button(label="Reset")
        reset_btn.connect("clicked", self.reset_values)
        button_hbox.pack_start(reset_btn, True, True, 0)

        self.start_btn = Gtk.Button(label="Start Stream")
        self.start_btn.connect("clicked", self.start_stream)
        button_hbox.pack_start(self.start_btn, True, True, 0)

        self.stop_btn = Gtk.Button(label="Stop Stream")
        self.stop_btn.connect("clicked", self.stop_stream)
        self.stop_btn.set_sensitive(False)
        button_hbox.pack_start(self.stop_btn, True, True, 0)

        vbox.pack_start(button_hbox, False, False, 0)

        self.update_filter_display()

    def on_scroll(self, widget, event):
        current_value = widget.get_value()
        increment = widget.get_adjustment().get_step_increment()

        if event.direction == Gdk.ScrollDirection.UP:
            widget.set_value(current_value + increment)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            widget.set_value(current_value - increment)

        return True

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

            try:
                with self.params_lock:
                    self.source_image = Image.open(self.image_path)
            except:
                pass
        dialog.destroy()

    def update_scale(self, widget):
        with self.params_lock:
            self.scale = int(widget.get_value())
        self.scale_label.set_text("Scale: %dpx" % self.scale)
        self.update_filter_display()

    def update_pan_x(self, widget):
        with self.params_lock:
            self.pan_x = int(widget.get_value())
        self.pan_x_label.set_text("Pan X: %dpx" % self.pan_x)
        self.update_filter_display()

    def update_pan_y(self, widget):
        with self.params_lock:
            self.pan_y = int(widget.get_value())
        self.pan_y_label.set_text("Pan Y: %dpx" % self.pan_y)
        self.update_filter_display()

    def update_phi(self, widget):
        with self.params_lock:
            self.phi = widget.get_value()
        self.phi_label.set_text("Phi (vertical tilt): %.2f°" % self.phi)
        self.update_filter_display()

    def update_theta(self, widget):
        with self.params_lock:
            self.theta = widget.get_value()
        self.theta_label.set_text("Theta (horizontal tilt): %.2f°" % self.theta)
        self.update_filter_display()

    def update_distance(self, widget):
        with self.params_lock:
            self.distance = widget.get_value()
        self.distance_label.set_text("Distance: %.0f" % self.distance)
        self.update_filter_display()

    def update_bg_scale(self, widget):
        with self.params_lock:
            self.bg_scale = widget.get_value()
        self.bg_scale_label.set_text("Background Scale: %.0f%%" % self.bg_scale)
        self.update_filter_display()

    def reset_values(self, widget):
        with self.params_lock:
            self.scale = self.default_scale
            self.pan_x = self.default_pan_x
            self.pan_y = self.default_pan_y
            self.phi = self.default_phi
            self.theta = self.default_theta
            self.distance = self.default_distance
            self.bg_scale = self.default_bg_scale

        self.scale_slider.set_value(self.scale)
        self.pan_x_slider.set_value(self.pan_x)
        self.pan_y_slider.set_value(self.pan_y)
        self.phi_slider.set_value(self.phi)
        self.theta_slider.set_value(self.theta)
        self.distance_slider.set_value(self.distance)
        self.bg_scale_slider.set_value(self.bg_scale)

    def build_filter(self):
        center_x = (self.output_width - self.scale) / 2
        center_y = (self.output_height - self.scale) / 2

        pad_x = int(center_x + self.pan_x)
        pad_y = int(center_y + self.pan_y)

        return (
            f"scale={self.scale}:-1:force_original_aspect_ratio=1,"
            f"pad={self.output_width}:{self.output_height}:{pad_x}:{pad_y}"
        )

    def update_filter_display(self):
        filter_str = self.build_filter()
        self.filter_label.set_text("Filter: %s" % filter_str)

    def apply_perspective(self, img, phi, theta, distance):
        if phi == 0.0 and theta == 0.0:
            return img

        width, height = img.size

        phi_rad = numpy.radians(phi)
        theta_rad = numpy.radians(theta)

        corners_3d = numpy.array([
            [-width/2, -height/2, 0],
            [width/2, -height/2, 0],
            [width/2, height/2, 0],
            [-width/2, height/2, 0]
        ], dtype=float)

        cos_phi = numpy.cos(phi_rad)
        sin_phi = numpy.sin(phi_rad)
        rot_x = numpy.array([
            [1, 0, 0],
            [0, cos_phi, -sin_phi],
            [0, sin_phi, cos_phi]
        ])

        cos_theta = numpy.cos(theta_rad)
        sin_theta = numpy.sin(theta_rad)
        rot_y = numpy.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])

        rotation = rot_y @ rot_x

        rotated = corners_3d @ rotation.T

        rotated[:, 2] += distance

        projected = numpy.zeros((4, 2))
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for i in range(4):
            if rotated[i, 2] > 0:
                projected[i, 0] = rotated[i, 0] * distance / rotated[i, 2]
                projected[i, 1] = rotated[i, 1] * distance / rotated[i, 2]
                min_x = min(min_x, projected[i, 0])
                max_x = max(max_x, projected[i, 0])
                min_y = min(min_y, projected[i, 1])
                max_y = max(max_y, projected[i, 1])

        proj_width = max_x - min_x
        proj_height = max_y - min_y

        canvas_width = int(proj_width * 1.2)
        canvas_height = int(proj_height * 1.2)

        offset_x = (canvas_width - proj_width) / 2 - min_x
        offset_y = (canvas_height - proj_height) / 2 - min_y

        target_coords = []
        for i in range(4):
            target_coords.append((projected[i, 0] + offset_x, projected[i, 1] + offset_y))

        coeffs = self.find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            target_coords
        )

        transformed = img.transform((canvas_width, canvas_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

        return transformed

    def find_coeffs(self, source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = numpy.array(matrix, dtype=float)
        B = numpy.array(source_coords).reshape(8)
        res = numpy.linalg.solve(A, B)
        return tuple(res)

    def process_frame(self):
        if not self.source_image:
            return None

        with self.params_lock:
            scale = self.scale
            pan_x = self.pan_x
            pan_y = self.pan_y
            phi = self.phi
            theta = self.theta
            distance = self.distance
            bg_scale = self.bg_scale

        try:
            bg = Image.open('wood-bg.jpg').convert('RGB')
            scale_factor = bg_scale / 100.0
            new_bg_width = int(bg.width * scale_factor)
            new_bg_height = int(bg.height * scale_factor)
            bg = bg.resize((new_bg_width, new_bg_height), Image.LANCZOS)
        except:
            bg = Image.new('RGB', (3840, 2160), (0, 0, 0))

        bg_width, bg_height = bg.size

        img = self.source_image.copy()
        aspect = img.width / img.height
        scaled_width = scale
        scaled_height = int(scale / aspect)
        img = img.resize((scaled_width, scaled_height), Image.LANCZOS)

        center_x = (bg_width - img.width) // 2
        center_y = (bg_height - img.height) // 2
        paste_x = center_x + pan_x
        paste_y = center_y + pan_y

        bg.paste(img, (paste_x, paste_y))

        if phi != 0.0 or theta != 0.0:
            scene = self.apply_perspective(bg, phi, theta, distance)
        else:
            scene = bg

        crop_x = (scene.width - self.output_width) // 2
        crop_y = (scene.height - self.output_height) // 2
        output = scene.crop((crop_x, crop_y, crop_x + self.output_width, crop_y + self.output_height))

        return output.convert('RGB')

    def stream_loop(self):
        cmd = [
            "ffmpeg",
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

        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        frame_time = 1.0 / 30.0

        while self.streaming:
            start = time.time()

            frame = self.process_frame()
            if frame:
                try:
                    self.ffmpeg_process.stdin.write(frame.tobytes())
                    self.ffmpeg_process.stdin.flush()
                except:
                    break

            elapsed = time.time() - start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()

    def start_stream(self, widget):
        self.image_path = self.file_entry.get_text()
        self.output_device = self.device_entry.get_text()

        try:
            self.source_image = Image.open(self.image_path)
        except:
            return

        self.streaming = True
        self.stream_thread = threading.Thread(target=self.stream_loop)
        self.stream_thread.start()

        self.start_btn.set_sensitive(False)
        self.stop_btn.set_sensitive(True)

    def stop_stream(self, widget=None):
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            self.stream_thread = None

        self.ffmpeg_process = None
        self.start_btn.set_sensitive(True)
        self.stop_btn.set_sensitive(False)

    def on_destroy(self, widget):
        self.stop_stream()
        Gtk.main_quit()


if __name__ == "__main__":
    window = PanZoomUI()
    window.show_all()
    Gtk.main()