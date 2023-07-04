#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


"""
This is an example of the kind of script you'd need.
This script resizes images in a folder using crop and pad; adapt to your dataset.
"""

import os
import shutil

from PIL import Image

_root = "/path/to/dataset"
size = 256
crop_sz = 224


for root, dirs, files in os.walk(_root):
    if "labels.txt" in files:
        for file in files:
            img_path = os.path.join(root, file)
            while img_path.endswith(".old"):
                shutil.move(img_path, f"{img_path[:-4]}")
                img_path = f"{img_path[:-4]}"
            moved_path = os.path.join(root, file + ".old")
            shutil.move(img_path, moved_path)
            img = Image.open(moved_path)
            w, h = img.size
            if w == 224:
                continue
            new_size = size, size
            if w > h:
                new_size = (size, h * size // w)
            new_img = img.resize(new_size)
            width, height = new_img.size  # Get dimensions
            left = (width - crop_sz) // 2
            top = (height - crop_sz) // 2
            right = (width + crop_sz) // 2
            bottom = (height + crop_sz) // 2

            # Crop the center of the image
            new_img = new_img.crop((left, top, right, bottom))
            new_img.save(img_path)
            img.close()
            new_img.close()
            os.remove(moved_path)
