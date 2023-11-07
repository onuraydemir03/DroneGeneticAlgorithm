import copy
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import cascaded_union

from utils.visualizer import Visualizer


class Point:
    def __init__(self,
                 x: int,
                 y: int):
        self.x = int(x)
        self.y = int(y)

    def distance_from(self, other_point: 'Point', method: str = "euclidian") -> int:
        """
        Calculates the distance between two points
        Args:
            other_point (Point): point that will be measured distance from this point
            method (str): optional distance calculation method['euclidian', 'manhattan']
        Returns:
            distance (int): measured distance of given point from this point
        """
        if method == "euclidian":
            dist = ((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2) ** .5
        elif method == "manhattan":
            dist = abs(self.x - other_point.x) + abs(self.y - other_point.y)
        else:
            raise TypeError("Valid methods are 'euclidian' & 'manhattan'. "
                            "Please fill the method parameter with a valid method name.")
        return int(dist)

    @staticmethod
    def mid_point_of(point: 'Point', other_point: 'Point') -> 'Point':
        """
        Calculates the mid-point of given two points
        Args:
            point (Point): 1. Point
            other_point (Point): 2. Point
        Returns:
            mid_point (Point): Mid-point of the two given points
        """
        x = (point.x + other_point.x) // 2
        y = (point.y + other_point.y) // 2
        return Point(x=x, y=y)

    def within(self, bbox: 'BoundingBox'):
        """
        Controls if a point is inside a bounding box
        Args:
            bbox (BoundingBox): BoundingBox that will be controlled
        Returns:
            is_inside (bool): True if this point is inside the given bbox False else
        """
        return bbox.xmin <= self.x <= bbox.xmax and bbox.ymin <= self.y <= bbox.ymax

    def within_polygon(self, polygon: 'Polygon'):
        """
        Controls if a point is inside a polygon
        Args:
            polygon (Polygon): Polygon that will be controlled
        Returns:
            is_inside (bool): True if this point is inside the given polygon False else
        """
        return polygon.geometric.intersects(ShapelyPoint(self.x, self.y))

    def as_list(self) -> List:
        """
        Returns the list representation of point as [x, y]
        """
        return [self.x, self.y]

    def as_tuple(self) -> Tuple:
        """
        Returns the tuple representation of point as (x, y)
        """
        return (self.x, self.y)

    def to_dict(self) -> Dict:
        """
        Returns the dict representation of point as {'x': x, 'y': y}
        """
        return {'x': self.x,
                'y': self.y}

    @staticmethod
    def from_dict(dict_obj) -> 'Point':
        """
        Parses the dict and converts it into Point class object
        Args:
            dict_obj (Dict): Dictionary object that will be parsed
        Returns:
            point (Point): Point object that is obtained from the source dictionary
        """
        required_params = ['x', 'y']
        for param in required_params:
            if param not in dict_obj.keys():
                raise ValueError(
                    f"There has been an error while creating class, provide required {required_params} parameters.")
        return Point(x=dict_obj.get('x'),
                     y=dict_obj.get('y'))

    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Polygon:
    def __init__(self,
                 points: List[Point]):
        if type(points[0]) != Point:
            points = list(map(lambda p: Point(*p), points))
        self.points = points

    def __len__(self) -> int:
        """
        Returns the number of points on this polygon
        """
        return len(self.points)

    @property
    def center(self) -> Point:
        """
        Returns the center of this polygon
        """
        center_point = self.geometric.centroid
        return Point(x=center_point.x, y=center_point.y)

    @property
    def xcenter(self) -> int:
        """
        Returns the x value in the center of this polygon
        """
        return self.center.x

    @property
    def ycenter(self) -> int:
        """
        Returns the y value in the center of this polygon
        """
        return self.center.y

    @property
    def sorted_points(self) -> List[Point]:
        """
        Returns the sorted points of this polygon by counterclockwise order and starting from top left point
        """
        return sort_coordinates_counterclockwise(self.points, self.center)

    @property
    def rect(self) -> 'BoundingBox':
        """
        Returns bounding box representation of Polygon
        """
        contours = np.float32(self.get_points_list())
        xmin, ymin, w, h = cv2.boundingRect(contours)
        xmax = xmin + w
        ymax = ymin + h
        return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @property
    def geometric(self) -> ShapelyPolygon:
        """
        Returns shapely.geometric.polygon representation of this polygon
        """
        return ShapelyPolygon(self.get_points_tuple())

    def merge_with_bbox(self, other: 'BoundingBox', force: bool = True) -> 'Polygon':
        """
        Merges this polygon with given bounding box,
        controls intersection of these two if control_intersection parameter is True and
        these two will not be merged if there is no intersection between them
        Args:
            other (BoundingBox): bbox that will be merged into this polygon
            force (bool): Controls these two bboxes are intersected or not
        Returns:
            merged_polygon (Polygon): polygon that is merged with given bounding box
        """
        bbox_polygon = Polygon(other.corners)
        return self.merge(other=bbox_polygon, force=force)

    def merge(self, other: 'Polygon', force: bool = True) -> 'Polygon':
        """
        Merges this polygon with given polygon,
        controls intersection of these two if control_intersection parameter is True and
        these two will not be merged if there is no intersection between them
        Args:
            other (Polygon): polygon that will be merged into this polygon
            force (bool): Controls these two bboxes are intersected or not
        Returns:
            merged_polygon (Polygon): polygon that is merged with given polygon
        """
        if force:
            intersection_area = self.get_intersection_area(other)
            if intersection_area == 0:
                raise ValueError("These two polygons are never intersects with each other."
                                 "If it is wanted to merge these two,"
                                 " it could be done by control_intersection=False parameter.")
            merged_polygon = self.geometric.union(other.geometric)
            points = list(map(lambda el: Point(*el), list(merged_polygon.exterior.coords)))
            return Polygon(points=points)
        merged_polygon = cascaded_union([self.geometric, other.geometric]).convex_hull
        points = list(map(lambda el: Point(*el), list(merged_polygon.exterior.coords)))
        return Polygon(points=points)

    def get_points_list(self) -> List:
        """
        Returns the list representation of all points as [[x1, y1], [x2, y2], ..., [xn, yn]]
        """
        return list(map(lambda point: point.as_list(), self.points))

    def get_points_tuple(self) -> Tuple:
        """
        Returns the tuple representation of all points as ((x1, y1), (x2, y2), ..., (xn, yn))
        """
        return tuple(map(lambda point: point.as_tuple(), self.points))

    def get_area(self) -> int:
        """
        Returns the area of this polygon
        """
        return self.geometric.area

    def get_intersection_area(self, another: 'Polygon') -> int:
        """
        Returns the intersected area of this polygon with given polygon
        Args:
            another (Polygon): polygon will be calculated intersection area
        Returns:
            intersection_area (int): intersection area of given polygon with this polygon
        """
        if not self.geometric.is_valid or not another.geometric.is_valid or not self.geometric.intersects(
                another.geometric):
            return 0
        return self.geometric.intersection(another.geometric).area

    def get_iou_score(self, another: 'Polygon') -> float:
        """
        Returns iou score of the given polygon with this polygon
        Args:
            another (Polygon): polygon will be calculated iou score
        Returns:
            iou_score (float): iou score of given polygon with this polygon
        """
        return self.get_intersection_area(another=another) / self.geometric.union(another.geometric).area

    def get_inside_rate(self, another: 'Polygon') -> float:
        """
        Returns inside rate of this polygon into given polygon
        Args:
            another (Polygon): polygon will be calculated inside rate
        Returns:
            inside_rate (float): inside rate of this polygon into given polygon
        """
        return self.get_intersection_area(another=another) / self.geometric.area

    def get_mask(self, image_height: int, image_width: int) -> 'Mask':
        """
        Returns boolean mask representation of this polygon on an image
        Args:
            image_height (int): Image height of mask
            image_width (int): Image width of mask
        Returns:
            mask (Mask): mask representation of this polygon
        """
        mask = Image.new('L', (image_width, image_height), color=0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(self.get_points_tuple(), fill=255)
        return Mask(np.array(mask, dtype=np.uint8))

    def to_dict(self) -> Dict:
        """
        Returns the dict representation of polygon as [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, ..., {'x': xn, 'y': yn}]
        """
        return {
            "points": [point.to_dict() for point in self.points]
        }

    @staticmethod
    def from_dict(dict_obj) -> 'Polygon':
        """
        Parses the dict and converts it into Polygon class object
        Args:
            dict_obj (Dict): Dictionary object that will be parsed
        Returns:
            polygon (Polygon): Polygon object that is obtained from the source dictionary
        """
        required_params = ['points']
        for param in required_params:
            if param not in dict_obj.keys():
                raise ValueError(
                    f"There has been an error while creating class, provide required {required_params} parameters.")
        points = list(map(lambda p_dict: Point.from_dict(dict_obj=p_dict), dict_obj.get('points')))
        return Polygon(points=points)

    def __eq__(self, other: 'Polygon') -> bool:
        return self.points == other.points

    def __repr__(self):
        point_str = " ".join(list(map(lambda point: point.__repr__(), self.points)))
        return f"Polygon({point_str})"


class Tetragon(Polygon):
    def __init__(self, points: List[Point]):
        if len(points) == 4:
            super().__init__(points=points)
        else:
            points = self.__map_to_four_points(Polygon(points=points))
            if len(points) == 4:
                super().__init__(points=points)
            else:
                raise ValueError(f"There has to be 4 points for a valid Tetragon, but your Polygon has {len(points)}")

    def __map_to_four_points(self, polygon: Polygon) -> List[Point]:
        """
        Creates a Tetragon object by reducing the number of points into 4
        Args:
            polygon (Polygon): polygon that has many points
        Returns:
            tetragon (Tetragon): Tetragon that have 4 points as a rotated rectangle
        """
        contours = np.expand_dims(np.float32(polygon.get_points_list()), axis=1)
        points = []
        for eps in np.linspace(0.001, 0.05, 20):
            epsilon = eps * cv2.arcLength(contours, True)
            points = cv2.approxPolyDP(contours, epsilon, True)  # approximate contours into four points
            if len(points) == 4:
                break
        points = np.squeeze(points).tolist()
        return list(map(lambda lst: Point(*lst), points))

    @staticmethod
    def from_bounding_box(bbox: 'BoundingBox'):
        """
        Creates a Tetragon class object from bounding box
        Args:
            bbox (BoundingBox): BoundingBox class object
        Returns:
            tetragon (Tetragon): Tetragon class object
        """
        return Tetragon(points=bbox.corners)

    @property
    def sorted_points(self) -> List[Point]:
        """
        Returns sorted Tetragon Point(x, y) pairs
        Returns:
            sorted_points (List[Point]): [top_left, bottom_left, bottom_right, top_right]
        """
        sorted_points = sort_coordinates_counterclockwise(points=self.points, center=self.center)
        list_of_sorted_points = list(map(lambda pnt: pnt.as_list(), sorted_points))
        roll_index = np.argsort(np.sum(list_of_sorted_points, axis=1))[0]
        if roll_index > 0:  # If roll_index is bigger than 0, it shows that top_left point is not on the top of the list
            sorted_points = np.roll(sorted_points, -roll_index).tolist()
        return sorted_points

    @property
    def top_left(self) -> Point:
        """
        Returns top left Point(x, y) corner of Tetragon
        """
        return self.sorted_points[0]

    @property
    def bottom_left(self) -> Point:
        """
        Returns bottom left Point(x, y) corner of Tetragon
        """
        return self.sorted_points[1]

    @property
    def bottom_right(self) -> Point:
        """
        Returns bottom right Point(x, y) corner of Tetragon
        """
        return self.sorted_points[2]

    @property
    def top_right(self) -> Point:
        """
        Returns top right Point(x, y) corner of Tetragon
        """
        return self.sorted_points[3]

    @property
    def top_mid(self) -> Point:
        """
        Returns mid-point(x, y) of top line
        """
        return Point.mid_point_of(self.top_left, self.top_right)

    @property
    def bottom_mid(self) -> Point:
        """
        Returns mid-point(x, y) of bottom line
        """
        return Point.mid_point_of(self.bottom_left, self.bottom_right)

    @property
    def left_mid(self) -> Point:
        """
        Returns mid-point(x, y) of left line
        """
        return Point.mid_point_of(self.top_left, self.bottom_left)

    @property
    def right_mid(self) -> Point:
        """
        Returns mid-point(x, y) of right line
        """
        return Point.mid_point_of(self.top_right, self.bottom_right)

    @property
    def width(self) -> int:
        """
        Returns width of tetragon (distance of right and left mid-points of lines)
        """
        return self.left_mid.distance_from(self.right_mid)

    @property
    def height(self) -> int:
        """
        Returns height of tetragon (distance of top and bottom mid-points of lines)
        """
        return self.top_mid.distance_from(self.bottom_mid)

    def ideal_world_coordinates(self):
        """
        This property holds ideal coordinates of tetragon according to its center & width & height values
        Returns:
            bbox (BoundingBox): ideal bounding box coordinates of Tetragon
        """
        center_point = self.center
        ideal_bbox_true = BoundingBox(
            xmin=center_point.x - (self.width // 2),
            ymin=center_point.y - (self.height // 2),
            xmax=center_point.x + (self.width // 2),
            ymax=center_point.y + (self.height // 2))
        # todo delete before PR
        diff = ideal_bbox_true.xmin - 50
        ideal_bbox_true.xmin -= diff
        ideal_bbox_true.xmax = ideal_bbox_true.xmin + 1100
        return ideal_bbox_true

    def calculate_transformation_matrix(self) -> np.ndarray:
        """
        Calculates transformation matrix of tetragon to map an ideal bounding box
        Args:
            src (list): source coordinates
            dst (list): destination coordinates
        Returns:
            transformation_matrix(np.ndarray): Transformation matrix that returns dst when A is given
        """
        src_list = list(map(lambda pnt: pnt.as_list(), self.sorted_points))
        dst_list = list(map(lambda pnt: pnt.as_list(), self.ideal_world_coordinates().corners))
        return cv2.getPerspectiveTransform(src=np.float32(src_list),
                                           dst=np.float32(dst_list))

    def __eq__(self, other: 'Tetragon') -> bool:
        return self.top_left == other.top_left and self.bottom_left == other.bottom_left \
            and self.bottom_right == other.bottom_right and self.top_right == other.bottom_right

    def __repr__(self):
        point_str = " ".join(list(map(lambda point: point.__repr__(), self.points)))
        return f"Tetragon({point_str})"


def sort_coordinates_counterclockwise(points: List[Point], center: Point) -> List[Point]:
    """
        Sorts Polygon points counterclockwise order, this sorting should be started from top left corner
        Args:
            points (list[Point]): Point(x,y) pairs that will be sorted by counterclockwise
            center (Point): Center point of points
        Returns:
            Sorted Point(x,y) pairs by counterclockwise
    """

    def ccw_sort_key(point):
        dx = point.x - center.x
        dy = point.y - center.y
        angle = math.atan2(dy, dx)
        return angle
    sorted_points = list(reversed(sorted(points, key=ccw_sort_key)))
    roll_index = np.argmax(list(map(lambda pnt: (center.x - pnt.x + center.y - pnt.y), sorted_points)))
    return np.roll(sorted_points, -roll_index).tolist()


class BoundingBox:
    def __init__(self,
                 xmin: int,
                 ymin: int,
                 xmax: int,
                 ymax: int):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    @classmethod
    def create(cls, center: Point, width: int, height: int):
        xmin = center.x - (width // 2)
        ymin = center.y - (height // 2)
        xmax = center.x + (width // 2)
        ymax = center.y + (height // 2)
        return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @property
    def left(self) -> int:
        """
        Returns left coordinate of bounding box
        """
        return min(self.xmin, self.xmax)

    @property
    def right(self) -> int:
        """
        Returns right coordinate of bounding box
        """
        return max(self.xmin, self.xmax)

    @property
    def top(self) -> int:
        """
        Returns top coordinate of bounding box
        """
        return min(self.ymin, self.ymax)

    @property
    def bottom(self) -> int:
        """
        Returns bottom coordinate of bounding box
        """
        return max(self.ymin, self.ymax)

    @property
    def center(self) -> Point:
        """
        Returns the center of bounding box
        """
        return Point(x=self.xcenter, y=self.ycenter)

    @property
    def xcenter(self) -> int:
        """
        Returns x coordinate of the bounding box center point
        """
        return (self.xmin + self.xmax) // 2

    @property
    def ycenter(self) -> int:
        """
        Returns y coordinate of the bounding box center point
        """
        return (self.ymin + self.ymax) // 2

    @property
    def width(self) -> int:
        """
        Returns width of the bounding box
        """
        return self.right - self.left

    @property
    def height(self) -> int:
        """
        Returns height of the bounding box
        """
        return self.bottom - self.top

    @property
    def top_left(self) -> Point:
        """
        Returns top left point of the bounding box
        """
        return Point(x=self.left, y=self.top)

    @property
    def bottom_left(self) -> Point:
        """
        Returns bottom left point of the bounding box
        """
        return Point(x=self.left, y=self.bottom)

    @property
    def bottom_right(self) -> Point:
        """
        Returns bottom right point of the bounding box
        """
        return Point(x=self.right, y=self.bottom)

    @property
    def top_right(self) -> Point:
        """
        Returns top right point of the bounding box
        """
        return Point(x=self.right, y=self.top)

    @property
    def top_mid(self) -> Point:
        """
        Returns top mid-point of the bounding box
        """
        return Point.mid_point_of(self.top_left, self.top_right)

    @property
    def bottom_mid(self) -> Point:
        """
        Returns bottom mid-point of the bounding box
        """
        return Point.mid_point_of(self.bottom_left, self.bottom_right)

    @property
    def left_mid(self) -> Point:
        """
        Returns left mid-point of the bounding box
        """
        return Point.mid_point_of(self.top_left, self.bottom_left)

    @property
    def right_mid(self) -> Point:
        """
        Returns right mid-point of the bounding box
        """
        return Point.mid_point_of(self.top_right, self.bottom_right)

    @property
    def slope(self) -> float:
        """
        Compute the slope of the line
        Returns:
            (float): slope of the line
        """
        if self.left == self.right:
            return 0
        return self.height / self.width

    @property
    def diagonal(self) -> List[Point]:
        """
        Returns diagonal (Top Left-Right Bottom) Point(x,y) pairs of BoundingBox as [TopLeft, BottomRight] pair
        """
        return [self.top_left, self.bottom_right]

    @property
    def corners(self) -> List[Point]:
        """
        Returns corner Point(x,y) pairs of BoundingBox as [TopLeft, BottomLeft, BottomRight, TopRight] ordered
        """
        return [self.top_left, self.bottom_left, self.bottom_right, self.top_right]

    def get_pascal_voc_format(self) -> Tuple[int, int, int, int]:
        """
        Returns:
            (tuple): (xmin, ymin, xmax, ymax)
        """
        return self.left, self.top, self.right, self.bottom

    def get_yolo_format(self) -> Tuple[int, int, int, int]:
        """
        Returns:
            (tuple): (xcenter, ycenter, width, height)
        """
        return self.xcenter, self.ycenter, self.width, self.height

    def get_x_at_y(self, y) -> int:
        """
        Get x-coordinate at given y-coordinate
        Args:
            y (int): y-coordinate
        Returns:
            (int): x-coordinate
        """
        if self.left == self.right:
            return self.left
        return int(self.left + (y - self.top) * (1 / self.slope))

    def get_y_at_x(self, x) -> int:
        """
        Get y-coordinate at given x-coordinate
        Args:
            x (int): x-coordinate
        Returns:
            (int): y-coordinate
        """
        if self.top == self.bottom:
            return self.top
        return int(self.top + (x - self.left) * self.slope)

    def get_area(self) -> int:
        """
        Get area of the bounding box
        Returns:
            area (int): total area
        """
        return self.width * self.height

    def get_intersection_area(self, bbox: 'BoundingBox') -> int:
        """
        Compute intersection area with given BoundingBox object
        Args:
            bbox (BoundingBox): BoundingBox object
        Returns:
            area (int): intersection area
        """
        intersection_width = np.maximum(0, min(self.right, bbox.right) - max(self.left, bbox.left))
        intersection_height = np.maximum(0, min(self.bottom, bbox.bottom) - max(self.top, bbox.top))
        return intersection_width * intersection_height

    def get_iou_score(self, bbox: 'BoundingBox') -> float:
        """
        Compute intersection over union (IoU) score with given BoundingBox object
        Args:
            bbox (BoundingBox): BoundingBox object
        Returns:
            iou_score (float) [0.0, 1.0]: IoU score
        """
        area1 = self.get_area()
        area2 = bbox.get_area()
        intersect = self.get_intersection_area(bbox)
        return intersect / (area1 + area2 - intersect)

    def get_inside_rate(self, bbox: 'BoundingBox') -> float:
        """
        Compute inside rate with given BoundingBox object
        Args:
            bbox (BoundingBox): BoundingBox object
        Returns:
            inisde_rate (float) [0.0, 1.0]: inside rate
        """
        intersection_area = self.get_intersection_area(bbox=bbox)
        return intersection_area / self.get_area()

    def merge(self, other: 'BoundingBox', force: bool = True):
        """
        Merge two BoundingBox objects and return it
        Args:
            other (BoundingBox): BoundingBox object
            force (bool): Controls these two bboxes are intersected or not
        Returns:
            merged_bbox (BoundingBox): merged bounding box
        """
        if force:
            intersection_area = self.get_intersection_area(other)
            if intersection_area == 0:
                raise ValueError("These two bounding boxes are never intersects with each other."
                                 "If it is wanted to merge these two,"
                                 " it could be done by control_intersection=False parameter.")
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return BoundingBox(xmin=left, ymin=top, xmax=right, ymax=bottom)

    def add_margin(self, margin: int, height: int, width: int) -> 'BoundingBox':
        """
        Adds margin into bounding box according to its image dimensions and returns it
        Args:
            margin (int): margin pixel value
            height (int): image height
            width (int): image width
        Returns:
            bbox (BoundingBox): bounding box that added margin
        """
        boundingBox = copy.deepcopy(self)
        boundingBox.xmin = max(0, boundingBox.xmin - margin)
        boundingBox.ymin = max(0, boundingBox.ymin - margin)
        boundingBox.xmax = min(width, boundingBox.xmax + margin)
        boundingBox.ymax = min(height, boundingBox.ymax + margin)
        return boundingBox

    def to_dict(self) -> Dict:
        """
        Returns the dict representation of bounding box as {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        """
        return {
            "xmin": round(float(self.xmin), 4),
            "ymin": round(float(self.ymin), 4),
            "xmax": round(float(self.xmax), 4),
            "ymax": round(float(self.ymax), 4)
        }

    @staticmethod
    def from_dict(dict_obj) -> 'BoundingBox':
        """
        Parses the dict and converts it into BoundingBox class object
        Args:
            dict_obj (Dict): Dictionary object that will be parsed
        Returns:
            bounding_box (BoundingBox): BoundingBox object that is obtained from the source dictionary
        """
        required_params = ['xmin', 'ymin', 'xmax', 'ymax']
        for param in required_params:
            if param not in dict_obj.keys():
                raise ValueError(
                    f"There has been an error while creating class, provide required {required_params} parameters.")
        return BoundingBox(xmin=dict_obj.get('xmin'),
                           ymin=dict_obj.get('ymin'),
                           xmax=dict_obj.get('xmax'),
                           ymax=dict_obj.get('ymax'))

    def __eq__(self, other: 'BoundingBox') -> bool:
        return self.xmin == other.xmin and self.ymin == other.ymin \
            and self.xmax == other.xmax and self.ymax == other.ymax

    def __repr__(self) -> str:
        return f'BoundingBox({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})'


class Mask:
    def __init__(self,
                 mask: np.ndarray):
        mask = mask.astype(np.uint8)
        self.mask = mask

    @property
    def contour(self) -> np.ndarray:
        """
        Get the largest contour array of the segmentation mask
        Returns:
            (np.ndarray): contour of the segmentation mask
        """
        contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        return contour

    @property
    def polygon(self) -> Polygon:
        """
        Calculates polygon representation of mask
        Returns:
            polygon (Polygon): polygon of mask
        """
        epsilon = 0.01 * cv2.arcLength(self.contour, True)
        polygon = cv2.approxPolyDP(self.contour, epsilon, True)
        points = np.squeeze(polygon, axis=1).tolist()
        return Polygon(points=points)

    def get_area(self) -> int:
        """
        Returns area of mask
        """
        return int(self.mask.astype(bool).sum())

    def get_intersection_area(self, mask: 'Mask') -> int:
        """
        Get intersection area of the given segmentation mask with this mask
        Args:
            mask (Mask): mask object
        Returns:
            (int): intersection area
        """
        intersection = int(np.logical_and(self.mask, mask.mask).astype(bool).sum())
        return intersection

    def get_union_area(self, mask: 'Mask') -> int:
        """
        Get the union area of the given segmentation mask with this mask
        Args:
            mask (Mask): mask object
        Returns:
            (int): union area
        """
        union = int(np.logical_or(self.mask, mask.mask).astype(bool).sum())
        return union

    def get_iou_score(self, mask: 'Mask') -> float:
        """
        Get intersection over union score of the given segmentation mask with this mask
        Args:
            mask (Mask): mask object
        Returns:
            (float): intersection over union score
        """
        intersection_area = self.get_intersection_area(mask)
        union_area = self.get_union_area(mask)
        return intersection_area / union_area

    def get_largest_bbox_of_mask(self) -> BoundingBox:
        """
        Get the largest bounding box of the segmentation mask
        Returns:
            bbox (BoundingBox): bbox of the segmentation mask
        """
        x1, x2 = np.where(np.any(self.mask, axis=0))[0][[0, -1]]
        y1, y2 = np.where(np.any(self.mask, axis=1))[0][[0, -1]]
        return BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)

    def get_largest_contour_bbox_of_mask(self) -> BoundingBox:
        """
        Get the largest contour array of the segmentation mask
        Returns:
            bbox (BoundingBox): bbox of the largest segmentation mask
        """
        x, y, w, h = cv2.boundingRect(self.contour)
        x1, y1, x2, y2 = x, y, x + w, y + h
        return BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)

    def get_mask_image_area_ratio(self) -> float:
        """
        Get the ratio of the segmentation mask area to the total image area
        Returns:
            (float): ratio of the segmentation mask area to the total image area
        """
        return np.sum(self.mask.astype(bool)) / (self.mask.shape[0] * self.mask.shape[1])

    def get_mask_bbox_area_ratio(self) -> float:
        """
        Get the ratio of the segmentation mask area to the rectangle of mask area
        Returns:
            (float): the ratio of the segmentation mask area to the rectangle of mask area
        """
        return np.sum(self.mask.astype(bool)) / self.get_largest_bbox_of_mask().get_area()

    def merge(self, other: 'Mask', force: bool = True) -> 'Mask':
        """
        Merge two segmentation masks
        Args:
            other (Mask): mask object
            force (bool): Controls these two bboxes are intersected or not
        Returns:
            merged_mask(Mask): merged segmentation mask
        """
        if force:
            intersection_area = self.get_intersection_area(other)
            if intersection_area == 0:
                raise ValueError("These two masks are never intersects with each other."
                                 "If it is wanted to merge these two,"
                                 " it could be done by control_intersection=False parameter.")
            return Mask(mask=np.logical_or(self.mask, other.mask))
        image = Visualizer.create(width=self.mask.shape[1], height=other.mask.shape[0]).image
        return self.polygon.merge(other.polygon).get_mask(image_height=image.height, image_width=image.width)

    def __eq__(self, other: 'Mask') -> bool:
        return self.get_iou_score(other) == 1.0

    def __repr__(self):
        return f"Mask (Area: {self.polygon.get_area()})"


class Entity:
    def __init__(self,
                 object_id: int,
                 score: float,
                 name: str):
        self.object_id = object_id
        self.score = score
        self.name = name

    def to_dict(self) -> Dict:
        """
        Returns the dict representation of entity
        """
        return {
            "object_id": int(self.object_id),
            "conf": float(self.score),
            "type": str(self.name),
        }

    @staticmethod
    def from_dict(dict_obj) -> 'Entity':
        """
        Parses the dict and converts it into Entity class object
        Args:
            dict_obj (Dict): Entity object that will be parsed
        Returns:
            entity (Entity): Entity object that is obtained from the source dictionary
        """
        required_params = ['object_id', 'conf', 'type']
        for param in required_params:
            if param not in dict_obj.keys():
                raise ValueError(
                    f"There has been an error while creating class, provide required {required_params} parameters.")
        return Entity(object_id=dict_obj.get('object_id'),
                      score=dict_obj.get('conf'),
                      name=dict_obj.get('type'))

    def __eq__(self, other: 'Entity') -> bool:
        return self.object_id == other.object_id and self.score == other.score and self.name == other.name

    def __repr__(self):
        return f"{self.name}(%{(self.score * 100):.1f})"


class Detection(BoundingBox, Entity):
    def __init__(self,
                 object_id: int,
                 xmin: int,
                 ymin: int,
                 xmax: int,
                 ymax: int,
                 score: float,
                 name: str):
        BoundingBox.__init__(self, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        Entity.__init__(self, object_id=object_id, score=score, name=name)

    def to_dict(self) -> Dict:
        """
        Returns the dict representation of detection
        """
        bbox_dict = BoundingBox.to_dict(self)
        entity_dict = Entity.to_dict(self)
        return {
            **entity_dict,
            **bbox_dict,
        }

    @staticmethod
    def from_dict(dict_obj) -> 'Detection':
        """
        Parses the dict and converts it into Detection class object
        Args:
            dict_obj (Dict): Detection object that will be parsed
        Returns:
            detection (Detection): Detection object that is obtained from the source dictionary
        """
        entity = Entity.from_dict(dict_obj)
        bbox = BoundingBox.from_dict(dict_obj)
        return Detection(**entity.__dict__, **bbox.__dict__)

    def __eq__(self, other: 'Detection') -> bool:
        return BoundingBox.__eq__(self, other) and Entity.__eq__(self, other)

    def __repr__(self):
        return Entity.__repr__(self)


class Segmentation(Mask, Entity):
    def __init__(self,
                 object_id: int,
                 mask: np.ndarray,
                 score: float,
                 name: str):
        Mask.__init__(self, mask=mask)
        Entity.__init__(self, object_id=object_id, score=score, name=name)

    def __repr__(self):
        return Entity.__repr__(self)
