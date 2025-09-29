#!/usr/bin/env python3

import sys
import subprocess
import threading
import time
import gi
gi.require_version('Gtk', '2.0')
gi.require_version('Gdk', '2.0')
from gi.repository import Gtk, Gdk, GObject
from PIL import Image
import io


class PanZoomUI(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="V4L2 Pan/Zoom Controller")
        self.set_default_size(600, 400)
        self.set_border_width(10)

        self.image_path = "sample.jpg"
        self.output_device = "/dev/video12"
        self.ffmpeg_process = None
        self.stream_thread = None
        self.streaming = False
        self.params_lock = threading.Lock()

        self.scale = 800
        self.pan_x = 0
        self.pan_y = 0
        self.output_width = 1920
        self.output_height = 1080

        self.source_image = None

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

        self.filter_label = Gtk.Label()
        self.filter_label.set_line_wrap(True)
        self.filter_label.set_alignment(0, 0.5)
        vbox.pack_start(self.filter_label, False, False, 0)

        button_hbox = Gtk.HBox(spacing=6)
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
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            self.file_entry.set_text(filename)
            self.image_path = filename
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

    def process_frame(self):
        if not self.source_image:
            return None

        with self.params_lock:
            scale = self.scale
            pan_x = self.pan_x
            pan_y = self.pan_y

        img = self.source_image.copy()

        aspect = img.width / img.height
        scaled_width = scale
        scaled_height = int(scale / aspect)

        img = img.resize((scaled_width, scaled_height), Image.LANCZOS)

        output = Image.new('RGB', (self.output_width, self.output_height), (0, 0, 0))

        center_x = (self.output_width - scaled_width) // 2
        center_y = (self.output_height - scaled_height) // 2

        paste_x = center_x + pan_x
        paste_y = center_y + pan_y

        output.paste(img, (paste_x, paste_y))

        return output.convert('RGB')

    def stream_loop(self):
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.output_width}x{self.output_height}",
            "-r", "30",
            "-i", "-",
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