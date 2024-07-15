
import cv2
import imageio
import matplotlib.pyplot as plt
import copy
# Own
from gaussian_renderer import render, network_gui
from utils.sh_utils import RGB2SH
from src.utility.graphic_utils import *




def render_rgb_and_depth(cam, gs_scene, pipe_settings, bg, debug=False):
    # Render rgb
    render_pkg = render(cam, gs_scene, pipe_settings, bg)
    net_image, depth_image = render_pkg["render"], render_pkg["depth"]

    rgb_image = net_image.cpu().permute((1, 2, 0))
    depth_image = depth_image.cpu().permute((1, 2, 0))

    if debug:
        depth_image2save = np.asarray((depth_image / depth_image.max() * 255).cpu()).astype('uint8')
        imageio.imwrite('image_depth.png', depth_image2save[..., 0])
        plt.imshow(depth_image2save)
        plt.show()

        rgb_image2save = np.asarray((rgb_image * 255).cpu()).astype('uint8')
        imageio.imwrite('image_rgb.png', rgb_image2save)
        plt.imshow(rgb_image2save)
        plt.show()

    return rgb_image, depth_image


def render_silhouette_mask(cam, gs_object_list, gs_env, width, height, color_set, pipe_settings, bg):
    semantic_gaussians_object_list = gs_object_list

    gaussian_scene_black = copy.deepcopy(gs_env)
    mask_env = torch.ones(gaussian_scene_black._xyz.shape[0], dtype=bool).to('cuda')
    mask_env[:gs_env._xyz.shape[0]] = False
    gaussian_scene_black.mask_points(mask_env)

    mask_silhouette = np.zeros((height, width, color_set.shape[0]))
    # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
    for gs_object_id in semantic_gaussians_object_list.keys():
        gaussian_scene_black_temp = copy.deepcopy(gaussian_scene_black)
        object_semantic_color = color_set[gs_object_id - 1]
        base_color = RGB2SH(object_semantic_color)
        current_object = semantic_gaussians_object_list[gs_object_id]
        current_object._features_dc[:] = current_object._features_dc_semantics
        current_object._features_rest[:, :] = current_object._features_rest_semantics

        gaussian_scene_black_temp.merge_gaussians(gaussian=current_object)

        # Render segmentation
        render_pkg = render(cam, gaussian_scene_black_temp, pipe_settings, bg)
        net_seg_image_silhoutte = render_pkg["render"]

        seg_silhouette_mask = net_seg_image_silhoutte.cpu().permute((1, 2, 0))

        distance = np.linalg.norm(seg_silhouette_mask.detach().cpu() - object_semantic_color.cpu().numpy(), axis=2)
        mask_silhouette[distance <= 0.1, gs_object_id - 1] = 1

    return mask_silhouette  # , bb_list


def render_visib_mask(cam, gs_environment, gs_object_list, color_set, height, width, pipe_settings, bg):
    gaussian_scene = copy.deepcopy(gs_environment)
    semantic_gaussians_object_list = gs_object_list

    # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
    for gs_object_id in semantic_gaussians_object_list.keys():
        object_semantic_color = color_set[gs_object_id - 1]
        current_object = semantic_gaussians_object_list[gs_object_id]
        current_object._features_dc[:] = current_object._features_dc_semantics
        current_object._features_rest[:, :] = current_object._features_rest_semantics
        gaussian_scene.merge_gaussians(gaussian=current_object)

    # Segmentation
    mask_env = torch.ones(gaussian_scene._xyz.shape[0], dtype=bool).to('cuda')
    mask_env[:gs_environment._xyz.shape[0]] = False
    gaussian_scene.mask_points(mask_env)

    # Render segmentation
    render_pkg = render(cam, gaussian_scene, pipe_settings, bg)
    net_seg_image = render_pkg["render"]

    seg_mask = net_seg_image.cpu().permute((1, 2, 0))
    invidiual_seg_masks = np.zeros((height, width, color_set.shape[0]))
    for c_i, c in enumerate(color_set):
        distance = np.linalg.norm(seg_mask - c.cpu().numpy(), axis=2)
        invidiual_seg_masks[distance <= 0.1, c_i] = 1  # to do change value!!

    seg_image = net_seg_image.cpu().permute((1, 2, 0))

    return invidiual_seg_masks, seg_image


def render_semanticsegmentation_mask(cam, gs_environment, gs_object_list, color_set, height, width, pipe_settings, bg,
                                     debug):
    gaussian_scene = copy.deepcopy(gs_environment)
    semantic_gaussians_object_list = gs_object_list

    # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
    for gs_object_id in semantic_gaussians_object_list.keys():
        current_object = semantic_gaussians_object_list[gs_object_id]
        current_object._features_dc[:] = current_object._features_dc_semantics
        current_object._features_rest[:, :] = current_object._features_rest_semantics
        gaussian_scene.merge_gaussians(gaussian=current_object)

    # Segmentation
    mask_env = torch.ones(gaussian_scene._xyz.shape[0], dtype=bool).to('cuda')
    mask_env[:gs_environment._xyz.shape[0]] = False
    gaussian_scene.mask_points(mask_env)

    # Render segmentation
    render_pkg = render(cam, gaussian_scene, pipe_settings, bg)
    net_seg_image = render_pkg["render"]

    seg_mask = net_seg_image.cpu()

    if debug:
        semantic_mask_image2save = (
                np.ascontiguousarray(seg_mask.permute((1, 2, 0))) * 255).astype(
            'uint8')
        cv2.imwrite('image_semantic_mask.png', semantic_mask_image2save)

    return (np.ascontiguousarray(seg_mask.permute((1, 2, 0))) * 255).astype('uint8')