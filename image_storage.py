from pathlib import Path
import os
import logging
import cv2
import numpy as np

class ImageStorage:

    logger = logging.getLogger('ImageStorage')
    storage_path = 'images'

    ### Crop settings
    img_height = 40
    img_width = 40
    img_pos_width = 112
    img_pos_height = 224


    def __init__(self, path='images'):
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise Exception(f'Path {path.resolve()} exists and is not a folder')
        elif not path.exists():
            os.makedirs(path.resolve(), exist_ok=True)
        self.storage_path = path.resolve()
        self.logger.info(f'Using {path.resolve()} as storage folder for images')


    def path(self, id):
        return f'{self.storage_path}/{id}'
    

    def exists(self, id):
        file_path = Path(self.path(id))
        return file_path.is_file()


    def crop(self, id, left, top, right, bottom):
        image = cv2.imread(self.path(id))

        shape_image = image[top:bottom, left:right]
        #cv2.imwrite('debug_shape.png', shape_image)
        shape_image = [cv2.resize(shape_image, (self.img_width, self.img_height))]
        shape_image = np.asarray(shape_image).reshape(1, self.img_height, self.img_width, 3)
        shape_image = (255. - shape_image) / 255.

        # Position [mirror effect for boxes close to the limits]
        image_height, image_width, _  = image.shape

        center_x = left + (right - left) / 2
        center_y = top + (bottom - top) / 2

        pos_left = int(max(0, center_x - self.img_pos_width / 2))
        pos_right = int(min(image_width, center_x + self.img_pos_width / 2))
        pos_top = int(max(0, center_y - self.img_pos_height / 2))
        pos_bottom = int(min(image_height, center_y + self.img_pos_height / 2))

        pad_left = int(abs(min(0, center_x - self.img_pos_width / 2)))
        pad_right = int(abs(min(0, image_width - (center_x + self.img_pos_width / 2))))
        pad_top = int(abs(min(0, center_y - self.img_pos_height / 2)))
        pad_bottom = int(abs(min(0, image_height - (center_y + self.img_pos_height / 2))))

        position_image = image[pos_top:pos_bottom, pos_left:pos_right]
        position_image = np.stack(
            [np.pad(position_image[:, :, c],
                    [(pad_top, pad_bottom), (pad_left, pad_right)],
                    mode='symmetric')
             for c in range(3)], axis=2)

        #cv2.imwrite('debug_position.png', position_image)

        position_image = np.asarray(position_image).reshape(1, self.img_pos_height, self.img_pos_width, 3)
        position_image = (255. - position_image) / 255.

        return (shape_image, position_image)
