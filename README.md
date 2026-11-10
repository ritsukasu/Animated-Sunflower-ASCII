<h1 align="center">Sunflower</h1>

<p align="center">
  A sunflower, grown one character at a time.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3-3776ab?logo=python&amp;logoColor=white">
  <img alt="pygame-ce" src="https://img.shields.io/badge/GUI-pygame--ce-4b8bbe">
  <img alt="Status: personal project" src="https://img.shields.io/badge/status-personal%20project-orange">
</p>

<p align="center">
  <img alt="An ASCII sunflower blooming in a desktop window" src="preview.gif" width="100%">
</p>

A small art project. A Python script opens a window and draws a sunflower out of plain text characters. It sprouts, blooms, and a ring of light sweeps across the seed head. It runs live, it's not a video.

## Quick start

```bash
git clone https://github.com/<your-username>/sunflower.git
cd sunflower
pip install -r requirements.txt
python sunflower.py
```

Press `SPACE` to replay the bloom, `S` to save a screenshot, `ESC` to quit.

## Project structure

```text
.
├── sunflower.py       # The whole artwork, one file
├── requirements.txt   # numpy, pygame-ce
└── preview.gif        # Animated preview for this README
```

## Built with

- [Python](https://www.python.org) for the script
- [NumPy](https://numpy.org) for the math behind every frame
- [pygame-ce](https://pyga.me) for the window and text drawing

## Credits

Inspired by the ASCII black hole by [@jin_22ee](https://github.com/JIN2-ASCII). Everything here is a small tribute to that idea.
