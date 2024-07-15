#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Built-in/Generic Imports
from typing import Tuple

# Built-in/Generic Imports
# ...

# Libs
import numpy as np
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion


# Own modules
# ...


class GPSVisualization(object):
    """
    Class for GPS data visualization using pre-downloaded OSM map in image format.
    Source: https://github.com/tisljaricleo/GPS-visualization-Python
    """

    def __init__(self, gps_data: dict, map_path: str, points: tuple):
        """
        Initialize the GPSVisualization class.

        Args:
            gps_data (dict): Raw GPS data.
            map_path (str): Path to pre-downloaded OSM map in image format.
            points (tuple): Upper-left and lower-right GPS points of the map (lat1, lon1, lat2, lon2).
        """
        gps_data_osm = []
        for key in gps_data.keys():
            gps_data_osm.extend(gps_data[key])

        self.gps_data = gps_data_osm
        self.points = points
        self.map_path = map_path

        self.result_image = Image
        self.x_ticks = []
        self.y_ticks = []

    def plot_map(self, output: str = 'save', save_as: str = 'resultMap.png', show: bool = False):
        """
        Plot or save the map image.

        Args:
            output (str, optional): Type 'plot' to show the map or 'save' to save it. Default is 'save'.
            save_as (str, optional): Name and type of the resulting image. Default is 'resultMap.png'.
            show (bool, optional): Whether to show the plotted map. Default is False.

        Returns:
            None
        """
        self.get_ticks()
        fig, axis1 = plt.subplots(figsize=(30, 30))
        axis1.imshow(self.result_image)
        axis1.set_xlabel('Longitude')
        axis1.set_ylabel('Latitude')
        axis1.set_xticks(list(self.x_ticks))
        axis1.set_yticks(list(self.y_ticks))
        axis1.set_xticklabels(self.x_ticks)
        axis1.set_yticklabels(self.y_ticks)
        axis1.grid()
        if output == 'save':
            plt.savefig(save_as)
        else:
            plt.show()

    def create_image(self, color, width=1):
        """
        Create an image with the original map and GPS records.

        Args:
            color (tuple): Color of the GPS records (RGB values).
            width (int, optional): Width of the drawn GPS records. Default is 1.

        Returns:
            None
        """
        # data = pd.read_csv(self.data_path, names=['LATITUDE', 'LONGITUDE'], sep=',')

        self.result_image = Image.open(self.map_path, 'r')
        img_points = []
        # gps_data = tuple(zip(data['LATITUDE'].values, data['LONGITUDE'].values))
        for lat, long, eval in self.gps_data:
            x1, y1 = self.scale_to_img((lat, long), (self.result_image.size[0], self.result_image.size[1]))
            img_points.append((x1, y1))
        draw = ImageDraw.Draw(self.result_image)
        # draw.line(img_points, fill=color, width=width)
        for point in img_points:
            draw.ellipse((point[0] - width, point[1] - width, point[0] + width, point[1] + width), fill=color)

    def scale_to_img(self, lat_lon: tuple, h_w: tuple) -> tuple:
        """
         Convert latitude and longitude to image pixels.

         Args:
             lat_lon (tuple): GPS record to draw (lat1, lon1).
             h_w (tuple): Size of the map image (width, height).

         Returns:
             tuple: x and y coordinates to draw on the map image.
         """
        # https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445
        old = (self.points[2], self.points[0])
        new = (0, h_w[1])
        y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        old = (self.points[1], self.points[3])
        new = (0, h_w[0])
        x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        # y must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        return int(x), h_w[1] - int(y)

    def get_ticks(self):
        """
        Generate custom ticks based on the GPS coordinates of the map for the matplotlib output.

        Returns:
            None
        """
        self.x_ticks = map(
            lambda x: round(x, 4),
            np.linspace(self.points[1], self.points[3], num=7))
        y_ticks = map(
            lambda x: round(x, 4),
            np.linspace(self.points[2], self.points[0], num=8))
        # Ticks must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        self.y_ticks = sorted(y_ticks, reverse=True)
