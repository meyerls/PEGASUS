import pylab as plt
from typing import Literal
import random

# Own Modules
from colmap_wrapper.gps.utils import *
from colmap_wrapper.gps.visualization import GPSVisualization


class GPSRegistration(object):
    def __init__(self):
        self.project_list: list = []

    def __gps_extract(self, debug: bool = False):
        """
        Extracts GPS data and extrinsic camera parameters from the project list.

        Args:
            debug (bool, optional): Whether to enable debug mode for intermediate visualizations. Default is False.

        Returns:
            None

        Note:
            - This function populates the `gps_data` and `extrinsic_data` attributes of the object.

        """
        gps_data = {}
        extrinsics = {}
        for model_idx, model in enumerate(self.project_list):
            gps_data.update({model_idx: []})
            extrinsics.update({model_idx: []})
            for image_idx in model.images.keys():
                gps_data[model_idx].append([float(model.images[image_idx].exifdata["XMP:GPSLatitude"]),
                                            float(model.images[image_idx].exifdata["XMP:GPSLongitude"]),
                                            float(model.images[image_idx].exifdata["XMP:AbsoluteAltitude"])])
                extrinsics[model_idx].append(model.images[image_idx].extrinsics)

        self.gps_data = {}

        for model_idx in gps_data.keys():
            # Show gps --> x and y
            gps_certesian = np.asarray(
                [convert_to_cartesian(lat, long, elev) for lat, long, elev in gps_data[model_idx]])
            if debug:
                plt.scatter(gps_certesian[:, 0], gps_certesian[:, 1])

            self.gps_data.update({model_idx: gps_certesian})

        if debug:
            plt.show()

        self.extrinsic_data = {}

        for model_idx in extrinsics.keys():
            # Show extrinsics (translational part)
            trans_certesian = np.asarray(
                [[array[:3, -1][0], array[:3, -1][1], array[:3, -1][2]] for array in extrinsics[model_idx]])
            if debug:
                plt.scatter(trans_certesian[:, 0], trans_certesian[:, 1])

            self.extrinsic_data.update({model_idx: trans_certesian})

        if debug:
            plt.show()

    def gps_visualize(self, map_path: str, osm_boarders: tuple, save_as: str, debug: bool = False):
        """
        Extracts GPS data and visualizes it on an OpenStreetMap (OSM) map image.

        Args:
            map_path (str): The path to the OSM map image.
            osm_boarders (tuple): The latitude and longitude boundaries of the map (min_lat, max_lat, min_lon, max_lon).
            save_as (str): The filename to save the final visualization.
            debug (bool, optional): Whether to enable debug mode for intermediate visualizations. Default is False.

        Returns:
            None

        Note:
            - OSM maps can be downloaded from: https://www.openstreetmap.org/export#map=5/33.174/14.590
            - For more information on visualizing GPS data on OSM maps, refer to: https://towardsdatascience.com/simple-gps-data-visualization-using-python-and-open-street-maps-50f992e9b676

        """
        self.__gps_extract(debug=debug)

        vis = GPSVisualization(gps_data=self.gps_data,
                               map_path=map_path,
                               points=osm_boarders)  # Draw converted records to the map image.

        vis.create_image(color=(0, 0, 255), width=0.2)
        vis.plot_map(show=debug, save_as=save_as)

    def coarse_align(self, debug: bool = False):
        """
        Performs coarse alignment of the point clouds using GPS data and extrinsic camera parameters.

        Args:
            debug (bool, optional): Whether to enable debug visualization. Default is False.

        Returns:
            None

        Raises:
            ValueError: If the number of sides in `project_list` is greater than 2.

        """
        if self.project_list.__len__() > 2:
            assert ValueError(
                'Up to know it is only possible to align two sides. Currently {} sides are present'.format(
                    self.project_list.__len__()))

        # Translate gps data to origin
        gps_offset = self.gps_data[0].mean(axis=0)

        # gps_data_origin = np.copy(self.gps_data)
        for model_idx in self.gps_data:
            for gps_idx, _ in enumerate(self.gps_data[model_idx]):
                self.gps_data[model_idx][gps_idx] -= gps_offset

        transformation_list = []

        for gps_idx, extrinsics_idx in zip(self.gps_data, self.extrinsic_data):
            R, c, t = kabsch_umeyama(self.gps_data[gps_idx], self.extrinsic_data[extrinsics_idx])
            transformation_list.append([R, c, t])

        if debug:
            colors = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)]
            pcd = []
            for gps_idx in self.gps_data.keys():
                # Plot GPS
                gps_pcd = points2pcd((np.asarray(self.gps_data[gps_idx])))
                gps_pcd.paint_uniform_color(random.choice(colors))
                pcd.append(gps_pcd)

            o3d.visualization.draw_geometries(pcd)

        if debug:
            colors = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)]
            gps_pcd_list = []
            extrinsics_pcd_list = []
            for idx in self.gps_data.keys():
                # Plot GPS
                color = random.choice(colors)

                gps_pcd = points2pcd((np.asarray(self.gps_data[idx])))
                gps_pcd.paint_uniform_color(color)
                gps_pcd_list.append(gps_pcd)

                # Plot Camera
                extrinsic_pcd = points2pcd((np.asarray(self.extrinsic_data[idx])))
                extrinsic_pcd.paint_uniform_color(color)
                extrinsics_pcd_list.append(extrinsic_pcd)

            o3d.visualization.draw_geometries(gps_pcd_list)
            o3d.visualization.draw_geometries(extrinsics_pcd_list)

        for idx, (R, c, t) in enumerate(transformation_list):
            T = np.eye(4)
            T[:3, :3] = c * R
            T[:3, -1] = t

            self.project_list[idx].transform(T=T)
            self.project_list[idx].save()

        if debug:
            pcd = []
            for idx in range(len(self.project_list)):
                pcd.append(self.project_list[idx].dense)
            o3d.visualization.draw_geometries(pcd)

        if debug:
            pcd = []
            for idx in range(len(self.project_list)):
                pcd.append(self.project_list[idx].dense)
                pcd.append(points2pcd((np.asarray(self.gps_data[idx]))))
                pcd.append(points2pcd((np.asarray(self.extrinsic_data[idx]))))

            o3d.visualization.draw_geometries(pcd)

    def fine_align(self, max_correspondence_distance: float = 0.02,
                   init_transformation: np.ndarray = np.eye(4),
                   estimation_method: Literal['point2point', 'point2plane'] = 'point2point',
                   debug: bool = False):
        """
        Aligns two point clouds using the Iterative Closest Point (ICP) algorithm.

        Args:
            max_correspondence_distance (float, optional): The maximum correspondence distance for ICP. Default is 0.02.
            init_transformation (numpy.ndarray, optional): The initial transformation matrix. Default is the identity matrix.
            estimation_method (Literal['point2point', 'point2plane'], optional): The estimation method for transformation.
                Choose between 'point2point' and 'point2plane'. Default is 'point2point'.
            debug (bool, optional): Whether to visualize the aligned point clouds. Default is False.

        Returns:
            None

        Raises:
            ValueError: If the number of sides in `project_list` is greater than 2.
            ValueError: If the provided estimation method is unknown.

        """
        if self.project_list.__len__() > 2:
            assert ValueError(
                'Up to know it is only possible to align two sides. Currently {} sides are present'.format(
                    self.project_list.__len__()))

        if estimation_method == 'point2point':
            method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif estimation_method == 'point2plane':
            method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            raise ValueError(
                "Method {} is unknown; Choose between point2point and point2plane!".format(estimation_method))

        reg_p2p = o3d.pipelines.registration.registration_icp(source=self.project_list[0].dense,
                                                              target=self.project_list[1].dense,
                                                              max_correspondence_distance=max_correspondence_distance,
                                                              init=init_transformation,
                                                              estimation_method=method)
        if debug:
            print(reg_p2p)
            print("Transformation is:")
            print(reg_p2p.transformation)

        self.project_list[0].dense.transform(reg_p2p.transformation)
        if debug:
            o3d.visualization.draw_geometries([self.project_list[0].dense, self.project_list[1].dense])

    def align(self):
        self.coarse_align()
        self.fine_align()


if __name__ == '__main__':
    from colmap_wrapper.dataloader.loader import COLMAPLoader
    from colmap_wrapper.visualization import ColmapVisualization
    from colmap_wrapper import USER_NAME

    if True:
        path = "/home/{}/Documents/reco/23_04_24/01".format(USER_NAME)
        img_orig = "/home/{}/Documents/reco/23_04_24/01/01_Satin".format(USER_NAME)
    else:
        path = '/media/{}/Samsung_T5/For5G/reco/23_03_17/03'.format(USER_NAME)
        img_orig = '/media/{}/Samsung_T5/For5G/data/23_03_17/03'.format(USER_NAME)
    project = COLMAPLoader(project_path=path,
                           exif_read=True,
                           img_orig=img_orig,
                           dense_pc='cropped.ply')

    # project_vs = ColmapVisualization(dataloader=project, bg_color=np.asarray([0, 0, 0]))
    # project_vs.visualization(show_sparse=False, frustum_scale=0.4, image_type='image', point_size=0.001)

    project.gps_visualize(map_path='map.png',
                          osm_boarders=(49.66721, 11.32313, 49.66412, 11.32784),
                          save_as='./result_map.png')

    project.align()  # project.coarse_align() + project.fine_align()

    # project.fuse_projects()
    # project.save_project()

    project_vs = ColmapVisualization(colmap=project, bg_color=np.asarray([0, 0, 0]))
    project_vs.visualization(show_sparse=False, frustum_scale=0.4, image_type='image', point_size=0.001)
