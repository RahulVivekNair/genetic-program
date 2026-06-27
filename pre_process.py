from PIL import Image

SIZE = 128

img = Image.open("input/openai.png")

print(img.mode)   # Should print P

# Convert palette -> RGBA
img = img.convert("RGBA")

# White background
background = Image.new("RGBA", img.size, (255, 255, 255, 255))
img = Image.alpha_composite(background, img)

# Now grayscale
img = img.convert("L")

# Resize
img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)

# Save BEFORE thresholding
img.save("output/target.png")

# # Threshold
# img = img.point(lambda p: 255 if p > 128 else 0)

# img.save("target.png")