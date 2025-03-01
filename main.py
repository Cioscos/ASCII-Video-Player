from PIL import Image
import numpy as np


ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"

def rgb_to_ansi(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def image_to_ascii_color(image_path, new_width=100):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    image = image.resize((new_width, new_height))
    pixels = np.array(image)  # Convertiamo direttamente in un array numpy per velocità

    ascii_image = ""
    for row in pixels:
        for r, g, b in row:
            r, g, b = int(r), int(g), int(b)  # Convertiamo in int per evitare overflow
            brightness = (r + g + b) // 3
            char = ASCII_CHARS[brightness * (len(ASCII_CHARS) - 1) // 255]
            ascii_image += f"{rgb_to_ansi(r, g, b)}{char}"
        ascii_image += "\033[0m\n"

    return ascii_image

def main():
    ascii_image = image_to_ascii_color("./images/input.jpg", new_width=500)
    print(ascii_image)

if __name__ == '__main__':
    main()
