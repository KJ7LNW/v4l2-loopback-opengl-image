from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="v4l2-loopback-opengl-image",
    version="1.0.0",
    author="KJ7LNW",
    description="OpenGL-based virtual camera application that streams 3D-rendered scenes to v4l2loopback devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KJ7LNW/v4l2-loopback-opengl-image",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "Pillow",
        "PyOpenGL",
        "PyGObject",
    ],
    entry_points={
        "console_scripts": [
            "v4l2-opengl-image=v4l2_loopback_opengl_image.v4l2_gl:main",
        ],
    },
)