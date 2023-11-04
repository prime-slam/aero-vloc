#  Copyright (c) 2023, Laura Hulley, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import requests


from pathlib import Path


class MapDownloader:
    """
    A class that allows to download a map from GoogleMaps in tile format
    """

    def __init__(
        self,
        north_west_lat: float,
        north_west_lon: float,
        south_east_lat: float,
        south_east_lon: float,
        zoom: int,
        overlap_level: float,
        api_key: str,
        folder_to_save: Path,
    ):
        """
        :param north_west_lat: Latitude of the northwest point of the map
        :param north_west_lon: Longitude of the northwest point of the map
        :param south_east_lat: Latitude of the southeast point of the map
        :param south_east_lon: Longitude of the southeast point of the map
        :param zoom: Zoom level of the map
        :param overlap_level: Shows how much neighboring images overlap each other. Float between 0 and 1
        :param api_key: API key for Google Maps API
        :param folder_to_save: Path to save map
        """
        self.north_west_lat = north_west_lat
        self.north_west_lon = north_west_lon
        self.south_east_lat = south_east_lat
        self.south_east_lon = south_east_lon
        self.zoom = zoom
        self.overlap_level = overlap_level
        self.api_key = api_key
        self.folder_to_save = folder_to_save

        # Maximum allowed shape and scale
        self.img_height = 640
        self.img_width = 640
        self.scale = 2
        self.map_type = "satellite"

        # Magic constants
        self.map_height = 256
        self.map_width = 256
        self.x_scale = math.pow(2, zoom) / (self.img_width / self.map_width)
        self.y_scale = math.pow(2, zoom) / (self.img_height / self.map_width)

    def __lat_lon_to_point(self, lat, lon):
        x = (lon + 180) * (self.map_width / 360)
        y = (
            (
                1
                - math.log(
                    math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
                )
                / math.pi
            )
            / 2
        ) * self.map_height

        return x, y

    def __point_to_lat_lon(self, x, y):
        lon = x / self.map_width * 360 - 180

        n = math.pi - 2 * math.pi * y / self.map_height
        lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

        return lat, lon

    def __get_image_bounds(self, lat, lon):
        centre_x, centre_y = self.__lat_lon_to_point(lat, lon)

        south_east_x = centre_x + (self.map_width / 2) / self.x_scale
        south_east_y = centre_y + (self.map_height / 2) / self.y_scale
        bottom_right_lat, bottom_right_lon = self.__point_to_lat_lon(
            south_east_x, south_east_y
        )

        north_west_x = centre_x - (self.map_width / 2) / self.x_scale
        north_east_y = centre_y - (self.map_height / 2) / self.y_scale
        top_left_lat, top_left_lon = self.__point_to_lat_lon(north_west_x, north_east_y)

        return top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon

    def __get_lat_step(self, lat, lon):
        point_x, point_y = self.__lat_lon_to_point(lat, lon)

        stepped_point_y = point_y - (self.map_height / self.y_scale)
        new_lat, _ = self.__point_to_lat_lon(point_x, stepped_point_y)

        lat_step = lat - new_lat

        return lat_step

    def __request_image(self, lat, lon):
        center = str(lat) + "," + str(lon)
        url = (
            "https://maps.googleapis.com/maps/api/staticmap?center="
            + center
            + "&zoom="
            + str(self.zoom)
            + "&size="
            + str(self.img_width)
            + "x"
            + str(self.img_height)
            + "&key="
            + self.api_key
            + "&maptype="
            + self.map_type
            + "&scale="
            + str(self.scale)
        )
        return requests.get(url).content

    def download_map(self):
        start_corners = self.__get_image_bounds(
            self.north_west_lat, self.north_west_lon
        )
        lon_step = start_corners[3] - start_corners[1]

        lat = self.north_west_lat
        index = 0
        metadata_file = open(self.folder_to_save / "map_metadata.txt", "w")
        metadata_file.write(
            "filename top_left_lat top_left_lon bottom_right_lat bottom_right_lon\n"
        )
        while lat >= self.south_east_lat:
            lon = self.north_west_lon

            while lon <= self.south_east_lon:
                image = self.__request_image(lat, lon)
                filename = f"{str(index).zfill(4)}.png"
                with open(self.folder_to_save / filename, "wb") as image_file:
                    image_file.write(image)
                (
                    top_left_lat,
                    top_left_lon,
                    bottom_right_lat,
                    bottom_right_lon,
                ) = self.__get_image_bounds(lat, lon)
                metadata_file.write(
                    f"{filename} {top_left_lat} {top_left_lon} {bottom_right_lat} {bottom_right_lon}\n"
                )

                lon = lon + (lon_step * (1 - self.overlap_level))
                index += 1

            lat_step = self.__get_lat_step(lat, lon)
            lat = lat + (lat_step * (1 - self.overlap_level))
        metadata_file.close()
