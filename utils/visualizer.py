import os
from typing import Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFont

from shelf_vision.util.constants import DATA_DIR

FONT_FILE = os.path.join(DATA_DIR, 'asset', 'Lato-Black.ttf')
MASK_COLOR = (149, 117, 222)
BOX_COLOR = (179, 19, 18)
POLYGON_COLOR = (179, 19, 18)
POINT_COLOR = (179, 19, 18)
TEXT_COLOR = (238, 226, 222)


class DrawerError(Exception):
    pass


class Visualizer:
    """
    Visualizes the entities
    Creates & crops & pastes sub images
    Shows & saves the drawing
    """
    def __init__(self, image: [np.ndarray, Image],
                 window_name: str = "Drawing",
                 fullscreen: bool = False,
                 window_height: int = 1080,
                 window_width: int = 1920):
        """
        Args:
            image (np.ndarray | PIL.Image): Displaying image
            fullscreen (bool): Displaying will be made on fullscreen window or not flag
            window_height (int): display window height
            window_width (int): display window width
        """
        if type(image) == np.ndarray:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if type(image) != Image.Image:
            try:
                if len(image.mode) == 3:
                    image = image.convert('RGB')
                if len(image.mode) == 4:
                    image = image.convert('RGBA')
            except Exception as exc:
                raise TypeError(f"Drawings are enabled & tested only np.ndarray and PIL images: {exc}")

        self.image = image.copy()
        self.window_name = window_name
        self.fullscreen = fullscreen

        self.window_height = window_height
        self.window_width = window_width

        self._window_initialized = False

    def _initialize_window(self):
        """
        Initializes the display window
        """
        cv2.namedWindow(winname=self.window_name, flags=cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(winname=self.window_name, height=self.window_height, width=self.window_width)
        self._window_initialized = True

    def draw_point(self, point, color: [Tuple, str] = POINT_COLOR, label: str = None, rotation: int = -45,
                   width: int = 10):
        """
        Draws a point on image
        Args:
            point (Point): entities Point class object
            color (Tuple): Color of the point
            label (str): Tag of the point
        """
        try:
            draw_image = ImageDraw.Draw(self.image)
            draw_image.ellipse((point.x, point.y, point.x + width, point.y + width), fill=color)
            if label is not None:
                self._put_text(label=label, x=point.x, y=point.y, outline_color=color, rotation=rotation)
        except Exception as exc:
            raise DrawerError(f"Exception while drawing point: {exc}")
        return self

    def draw_line(self, from_point, to_point, color: [Tuple, str] = BOX_COLOR, line_width: int = 4, label: str = None, rotation: int = -45):
        """
        Draws a line on image
        Args:
            from_point (Point): start Point of line
            from_point (Point): end Point of line
            color (Tuple): Color of the point
            label (str): Tag of the point
        """
        try:
            draw_image = ImageDraw.Draw(self.image)
            draw_image.line((from_point.x, from_point.y, to_point.x, to_point.y), fill=color, width=line_width)
            if label is not None:
                self._put_text(label=label, x=from_point.x, y=from_point.y, outline_color=color, rotation=rotation)
        except Exception as exc:
            raise DrawerError(f"Exception while drawing line: {exc}")
        return self

    def draw_polygon(self, polygon, color: [Tuple, str] = POLYGON_COLOR, line_width: int = 4, label: str = None, rotation: int = -45,
                     background: bool = True):
        """
        Draws a polygon on image
        Args:
            polygon (Polygon): entities Polygon class object
            color (Tuple): Color of the polygon
            line_width (int): Width of the surrounding line
            label (str): Tag of the polygon
            rotation (int): Rotation of text
            mask(bool): mask the polygon background
        """
        try:
            contour = list(map(lambda pnt: pnt.as_tuple(), polygon.points))
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(contour, fill=255)
            roi = ImageChops.composite(self.image, Image.new('RGB', self.image.size), mask)
            if background:
                darkened_roi = roi.point(lambda p: p * 0.5)
                self.image = Image.composite(darkened_roi, self.image, mask)
            draw_image = ImageDraw.Draw(self.image)
            draw_image.polygon(contour, outline=color, width=line_width)
            if label is None:
                label = str(polygon)
            self._put_text(label=label, x=polygon.sorted_points[0].x, y=polygon.sorted_points[0].y, outline_color=color, rotation=rotation)
        except Exception as exc:
            raise DrawerError(f"Exception while drawing polygon: {exc}")
        return self

    def draw_bbox(self, bbox, color: [Tuple, str] = BOX_COLOR, line_width: int = 4, label: str = None, rotation: int = -45,
                  fill: bool = False):
        """
        Draws a box on image
        Args:
            bbox (BoundingBox): entities BoundingBox class object
            color (Tuple): Color of the bbox
            line_width (int): Width of the surrounding line
            label (str): Tag of the bbox
        """
        try:
            draw_image = ImageDraw.Draw(self.image)
            shape = bbox.get_pascal_voc_format()
            if fill:
                draw_image.rectangle(shape, fill=color, width=line_width)
            else:
                draw_image.rectangle(shape, outline=color, width=line_width)
            if label is None:
                label = str(bbox)
            self._put_text(label=label, x=bbox.left, y=bbox.top, outline_color=color, rotation=rotation)
        except Exception as exc:
            raise DrawerError(f"Exception while drawing rectangle: {exc}")
        return self

    def draw_mask(self, mask, color: [Tuple, str] = MASK_COLOR, label: str = None, rotation: int = -45):
        """
        Draws a mask on image
        Args:
            mask (Mask): entities Mask class object
            color (Tuple): Color of the mask
            label (str): Tag of the mask
        """
        try:
            mask_ = mask.mask.astype(np.float32)
            mask_ /= mask_.max()
            mask_ = np.stack([mask_, mask_, mask_], axis=2)

            image_np = np.array(self.image)
            color_rgb = np.array([color[0], color[1], color[2]])
            image_np = cv2.addWeighted(image_np, 1, (mask_ * color_rgb).astype(np.uint8), 0.4, 0)
            self.image = Image.fromarray(image_np.astype(np.uint8), mode='RGB')
            if label is None:
                label = str(mask_)
            self._put_text(label=label, x=mask.polygon.sorted_points[0].x, y=mask.polygon.sorted_points[0].y,
                           outline_color=color, rotation=rotation)
        except Exception as exc:
            raise DrawerError(f"Exception while drawing mask: {exc}")
        return self

    def _put_text(self, label: str, x: int, y: int, font_size: int = 16, text_color=TEXT_COLOR, outline_color=BOX_COLOR,
                  line_width: int = 4, rotation: int = -45, outline: bool = True):
        """
        Puts a text on image
        Args:
            label (str): Text will be putted on image
            x (int): X coordinate of the text
            y (int): Y coordinate of the text
            font_size (int): Font size of the text
            text_color (Tuple): Color of the text
            outline_color (Tuple): Outline rectangle color of the text bbox
            line_width (int): Width of the surrounding line of outline rectangle
        """
        try:
            self.font = ImageFont.truetype(FONT_FILE, font_size)
            draw_image = ImageDraw.Draw(self.image)
            text_w, text_h = draw_image.textsize(str(label), font=self.font)
            font_size = (text_w, text_h + 5)
            rototed_draw_image = Image.new('RGBA', font_size, (0, 0, 0, 0))
            rotated_draw = ImageDraw.Draw(rototed_draw_image)
            if outline:
                rect = [(0, 0), (text_w + 10, text_h + 10)]
                # noinspection PyTypeChecker
                rotated_draw.rectangle(rect, outline=outline_color, width=line_width, fill=outline_color)
            rotated_draw.multiline_text(xy=(0, 0),
                                        text=str(label),
                                        font=self.font,
                                        fill=text_color)
            rototed_draw_image = rototed_draw_image.rotate(rotation, expand=1)
            self.image.paste(rototed_draw_image, (int(x), int(y)), rototed_draw_image)
        except Exception as exc:
            raise DrawerError(f"Exception while putting text: {exc}")
        return self

    def put_sub_image(self, paste_image: Image, x: int, y: int):
        """
        Puts a sub image into the self-image from top_left x, y points
        Args:
            paste_image (Image): Sub image that will be pasted
            x (int): X coordinate of the top left corner on self-image
            y (int): Y coordinate of the top left corner on self-image
        """
        try:
            if len(paste_image.getbands()) == 4:
                self.image.paste(paste_image, (int(x), int(y)), paste_image)
            else:
                self.image.paste(paste_image, (int(x), int(y)))
        except Exception as exc:
            raise DrawerError(f"Exception while putting sub image: {exc}")

    @staticmethod
    def create(width: int, height: int, num_of_channels: int = 3, fill: Tuple = (255, 255, 255), window_name: str = "Creation"):
        """
        Creates an empty Image and returns its Visualizer object
        Args:
            width (int): width of the blank image
            height (int): height of the blank image
            num_of_channels (int): Type of the image L, RGB, RGBA
            fill (Tuple): filling color
        """
        if num_of_channels == 1:
            created_image = Image.new('L', (width, height), fill)
        elif num_of_channels == 3:
            created_image = Image.new('RGB', (width, height), fill)
        elif num_of_channels == 4:
            created_image = Image.new('RGBA', (width, height), fill)
        else:
            raise DrawerError(f"Exception while creating a blank image, "
                              f"num_of_channels is not valid (1, 2, 3): {num_of_channels}")
        return Visualizer(image=created_image, window_name=window_name)

    def save(self, output_path: str):
        """
        Saves the image to output_path
        Args:
            output_path (str): Save dir of the image
        """
        self.image.save(output_path)

    def show_image(self, delay: int = 1):
        """
        Shows the image in pre-initialized opencv-window
        If it is not initialized, it initializes and shows after that
        Args:
            delay (int): Milliseconds of the show window.
                0: Show and wait until press key
                1, 10, 500, X...: Wait X msec and continue
        """
        try:
            if not self._window_initialized:
                self._initialize_window()
            cv2.imshow(winname=self.window_name,
                       mat=cv2.cvtColor(np.array(self.image), cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(delay=delay)
            return key
        except Exception as exc:
            raise DrawerError(f"Exception while showing image: {exc}")

    def set_trigger(self, function, parameters):
        """
        Sets the mouseCallback trigger to the displaying window
        Args:
            function (Function*): Function that will be run with params
            parameters (Parameters*): Parameters that will be passed into function
        """
        if not self._window_initialized:
            self._initialize_window()
        cv2.setMouseCallback(self.window_name,
                             function,
                             parameters)
