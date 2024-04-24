import os
import random
import numpy as np
from PIL import Image

from bullet import SimImage

def get_aug_img(real_path):
    image_files = [f for f in os.listdir(real_path) if f.endswith(('jpg', 'jpeg', 'png', 'gif'))]
    # Choose a random image file
    random_image_file = random.choice(image_files)

    # Load the image using PIL
    image_path = os.path.join(real_path, random_image_file)
    image = Image.open(image_path)

    # Get the dimensions of the image
    width, height = image.size

    # Calculate the coordinates for cropping
    left = random.randint(0, width - 200)
    top = height - 200
    right = left + 200
    bottom = height

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Convert the cropped image to numpy array
    image_array = np.array(cropped_image)
    image_array = image_array[:, :, :3]
    return image_array

def generate_data(line_path, real_path, texture_path, line_dataset_size):
    sim = SimImage(texture_path)
    # Line Generate Loop
    for i in range(line_dataset_size):
        # Get an image + waypoints
        img_raw, waypoints = sim.generate_line_images()
        
        # Save mask
        img_alpha = img_raw[:, :, 3]
        img_blank = np.expand_dims(img_alpha, axis=-1)
        img_blank = np.repeat(img_blank, 3, axis=-1)
        mask = np.all(img_raw[:, :, :3] != img_blank, axis=-1)
        mask_img = Image.fromarray(mask)
        mask_img.save(f'{line_path}masks/mask_image_{i}.jpeg')

        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)

        anti_mask = np.all(img_raw[:, :, :3] == img_blank, axis=-1)
        anti_mask = np.expand_dims(anti_mask, axis=-1)
        anti_mask = np.repeat(anti_mask, 3, axis=-1)

        # Augment and save image
        aug_image = get_aug_img(real_path)
        img_rgb = (aug_image * anti_mask) + (img_raw[:, :, :3] * mask)
        img_rgb = Image.fromarray(img_rgb[:, :, :3])
        img_rgb.save(f'{line_path}images/line_image_{i}.jpeg')

        with open(f'{line_path}labels/point_label_{i}.txt', 'w') as f:
            f.write(str(waypoints[1].tolist())) # change index to get different waypoints
        f.close()