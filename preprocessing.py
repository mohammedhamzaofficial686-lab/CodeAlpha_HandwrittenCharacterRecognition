"""Shared normalization for training, uploads, and browser drawings."""
import io
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 12_000_000

def normalize_glyph(ink):
    """White foreground on black; preserve aspect ratio inside a 20px box."""
    ink = np.asarray(ink, dtype=np.uint8)
    ys, xs = np.where(ink > 30)
    if len(xs) < 3:
        raise ValueError("No clear character found. Draw with a thicker stroke.")
    crop = Image.fromarray(ink[ys.min():ys.max()+1, xs.min():xs.max()+1])
    scale = 20 / max(crop.size)
    size = tuple(max(1, round(v*scale)) for v in crop.size)
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28,28))
    canvas.paste(crop, ((28-size[0])//2, (28-size[1])//2))
    return np.asarray(canvas).copy()

def decode_image(raw):
    if not raw or len(raw) > 8*1024*1024:
        raise ValueError("Choose a PNG, JPEG, or WebP under 8 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Use PNG, JPEG, or WebP.")
            source = ImageOps.exif_transpose(source)
            if source.width*source.height > 12_000_000:
                raise ValueError("Image exceeds 12 megapixels.")
            rgba = source.convert("RGBA")
            paper = Image.new("RGBA", rgba.size, "white")
            paper.alpha_composite(rgba)
            gray = paper.convert("L")
            gray.thumbnail((1400,700))
            pixels = np.asarray(gray, dtype=np.float32)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("That image could not be opened.") from exc
    edges = np.concatenate((pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]))
    # Accept dark ink on light paper or light ink on a dark background.
    ink = 255-pixels if np.median(edges) >= 128 else pixels
    ink = np.clip(ink - np.percentile(ink,10), 0,255)
    if ink.max() < 25:
        raise ValueError("The image is blank or has too little contrast.")
    return (ink / ink.max() * 255).astype(np.uint8)

def segment(raw, expected=None):
    ink = decode_image(raw)
    mask = ink > max(45, float(ink.max())*0.23)
    # Projection keeps vertically detached dots of i/j with their stems.
    occupied = mask.sum(axis=0) >= max(1, ink.shape[0]//350)
    boundaries = []
    start = None
    for x, filled in enumerate(np.r_[occupied, False]):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            boundaries.append([start,x]); start = None
    if not boundaries:
        raise ValueError("No handwriting found.")
    # Merge tiny one-pixel cracks, but never force a requested character count.
    groups = []
    for left,right in boundaries:
        if groups and left-groups[-1][1] <= 1:
            groups[-1][1] = right
        else:
            groups.append([left,right])
    glyphs, boxes = [], []
    for left,right in groups:
        ys,xs = np.where(mask[:,left:right])
        if len(xs)<5:
            continue
        top,bottom = int(ys.min()),int(ys.max())+1
        glyphs.append(normalize_glyph(ink[top:bottom,left:right]))
        boxes.append([left,top,right,bottom])
    if not glyphs:
        raise ValueError("No clear characters found.")
    if len(glyphs)>20:
        raise ValueError("Use at most 20 separated characters on one line.")
    if expected is not None and len(glyphs)!=expected:
        raise ValueError(f"Detected {len(glyphs)} characters, but you expected {expected}. Leave clear gaps or upload one character at a time.")
    return glyphs, boxes

def tensor_images(images):
    import torch
    a = np.stack(images).astype(np.float32)/255
    return torch.from_numpy(a[:,None,:,:])
