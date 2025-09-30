# v4l2-loopback-opengl-image

OpenGL-based virtual camera application that streams 3D-rendered scenes to v4l2loopback devices via FFmpeg.

![Application Screenshot](screenshots/main-window.png)

## Description

v4l2-loopback-opengl-image creates a virtual camera by rendering a 3D scene with OpenGL and streaming it through FFmpeg to a [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback) device. While this application was tested with v4l2loopback, it should work with any V4L2 device that supports output mode (such as USB HDMI capture devices).

**Note:** Users are responsible for installing v4l2loopback if they wish to use it as a virtual camera device. See the Installation section below.

### Use Case: Web Browser Camera/Image Upload Workaround

Some websites require camera access for image capture but don't provide a file upload option. This tool solves that problem by presenting a static image through a virtual camera that browsers like Firefox and Chrome can access. Instead of being forced to use a physical camera or being unable to submit an image at all, you can load any image file and present it as a camera feed.

The 3D rendering environment provides additional capabilities:
- Position and orient the image naturally in 3D space
- Adjust perspective and viewing angle interactively
- Control lighting to ensure optimal visibility
- Pan and zoom to focus on specific areas of the image
- Present professional-looking content with anti-aliased rendering

Additional applications:
- Showing product images or barcodes for verification on camera-only websites
- Displaying technical diagrams or documents through browser-based camera interfaces
- Testing camera input applications with controlled 3D-rendered content
- Creating dynamic visual presentations with adjustable perspective

### Features

The application provides:

- **3D Scene**: Textured cube and card objects in an interactive 3D environment
- **Camera Controls**: Mouse-driven pan, zoom, and rotation with reset view
- **Anti-Aliasing**: 8x MSAA with sample shading, hardware dithering, and Bayer matrix dithering
- **Lighting**: Interactive light source control with spherical positioning
- **Device Selection**: Dropdown menu for selecting v4l2loopback output devices
- **Image Loading**: Load and display images on the 3D card with aspect ratio preservation
- **Output Control**: Configurable horizontal flip and cube size adjustment

## Requirements

- Python 3
- GTK 3
- OpenGL 3.3+
- FFmpeg
- v4l2loopback kernel module (if using virtual camera)
- Python packages: PyOpenGL, numpy, Pillow, PyGObject

## Installation

### System Dependencies

```bash
# Install system dependencies (Debian/Ubuntu)
sudo apt-get install python3-gi python3-opengl ffmpeg

# Install Python packages
pip3 install numpy Pillow PyOpenGL
```

### v4l2loopback Installation

If you want to use this application as a virtual camera (recommended), you must install v4l2loopback:

```bash
# Debian/Ubuntu
sudo apt-get install v4l2loopback-dkms

# Or build from source
git clone https://github.com/v4l2loopback/v4l2loopback.git
cd v4l2loopback
make && sudo make install
sudo depmod -a
```

## Usage

```bash
# Load v4l2loopback module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Virtual Camera"

# Run the application
python3 v4l2-gl.py [/dev/videoX]
```

Optional command-line argument:
- `/dev/videoX`: Specify the v4l2loopback device to use (defaults to first available)

## Controls

### Mouse
- **Left-click + drag**: Rotate camera
- **Right-click + drag**: Rotate light source
- **Shift + left-click + drag**: Pan card position
- **Mouse wheel**: Zoom camera in/out

### UI Controls
- **Device dropdown**: Select v4l2loopback output device
- **Load Image**: Change the displayed image on the card
- **Cube Size slider**: Adjust cube dimensions (12-48 inches)
- **Flip Horizontal checkbox**: Mirror output horizontally
- **Reset View**: Restore default camera, light, and card positions

## Architecture

The application uses a multi-threaded design:
- **Main thread**: GTK UI and OpenGL rendering
- **Worker thread**: Frame timing and FFmpeg I/O
- **Communication**: Queue-based frame passing with thread-safe synchronization

Anti-aliasing stack:
1. 8x MSAA with multisample framebuffer
2. Per-sample fragment shading (GL_SAMPLE_SHADING)
3. Hardware dithering (GL_DITHER)
4. 8x8 Bayer matrix dithering in fragment shader
5. Trilinear texture filtering with mipmaps

## License

GPL-3.0
