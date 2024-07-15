import matplotlib.pyplot as plt
import open3d as o3d
import imageio
import numpy as np
import matplotlib.pyplot as plt
# np.concatenate([depth_image[..., None], depth_image[..., None], depth_image[..., None]], axis=2)
import cv2

if False:
    depth_image = imageio.imread('./000001_000000.depth.png')
    depth_image_o3d = o3d.geometry.Image(depth_image)

    K = np.asarray([[438.2178138639046, 0.0, 320.0],
                    [0.0, 492.56402851864937, 240.0],
                    [0.0, 0.0, 1.0]])
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, intrinsic_matrix=K)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_image_o3d,
                                                          intrinsic=intrinsics)
    pcd_cleaned, _ = pcd.remove_radius_outlier(50, 0.1)
    pcd_cleaned.estimate_normals()
    o3d.visualization.draw_geometries([pcd_cleaned])


def denoise_depth_map(depth_map):
    # Apply median filtering
    depth_map_filtered = cv2.medianBlur(depth_map, ksize=5)

    # Apply bilateral filtering
    depth_map_filtered = cv2.bilateralFilter(depth_map_filtered, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply non-local means denoising
    #depth_map_filtered = cv2.fastNlMeansDenoising(depth_map_filtered, h=10, templateWindowSize=7, searchWindowSize=21)

    return depth_map_filtered

def compute_normal_map(depth_map, smoothing=True, smoothing_kernel_size=5):
    # Ensure the depth map is a floating-point type
    depth_map_float = depth_map.astype(np.float32)

    # Apply Gaussian smoothing if required
    if smoothing:
        depth_map_float = cv2.GaussianBlur(depth_map_float, (smoothing_kernel_size, smoothing_kernel_size), 2)

    # Sobel operators for gradient in X and Y directions
    grad_x = cv2.Sobel(depth_map_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map_float, cv2.CV_32F, 0, 1, ksize=3)

    # Initialize normal map
    normal_map = np.zeros_like(depth_map_float, shape=(*depth_map_float.shape, 3))

    # Calculate normals
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            dzdx = grad_x[y, x]
            dzdy = grad_y[y, x]

            normal = np.array([-dzdx, -dzdy, 1])
            normal = normal / np.linalg.norm(normal)  # Normalize the vector

            normal_map[y, x] = normal

    # Convert normal values from [-1, 1] to [0, 1] for visualization
    normal_map_visual = (normal_map + 1) / 2 * 255
    normal_map_visual = normal_map_visual.astype(np.uint8)

    return normal_map, normal_map_visual


def compute_ssao(depth_map, kernel_size=5, occlusion_radius=15):
    height, width = depth_map.shape
    ssao_map = np.zeros_like(depth_map, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            occlusion_count = 0
            depth_center = depth_map[y, x]

            for ky in range(-kernel_size, kernel_size + 1):
                for kx in range(-kernel_size, kernel_size + 1):
                    ny, nx = y + ky, x + kx
                    if 0 <= ny < height and 0 <= nx < width:
                        depth_neighbor = depth_map[ny, nx]
                        if abs(depth_neighbor - depth_center) > occlusion_radius:
                            occlusion_count += 1

            ssao_map[y, x] = 1 - occlusion_count / (kernel_size * 2 + 1) ** 2

    return ssao_map

def generate_sample_kernels(kernel_size=16):
    kernels = []
    for _ in range(kernel_size):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)  # Only in the hemisphere
        kernel = np.array([x, y, z])
        kernel /= np.linalg.norm(kernel)  # Normalize
        kernels.append(kernel)
    return np.array(kernels)


def compute_ssao_with_random_kernels(depth_map, kernels, occlusion_radius=15):
    height, width = depth_map.shape
    ssao_map = np.zeros_like(depth_map, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            occlusion_count = 0
            depth_center = depth_map[y, x]

            for kernel in kernels:
                sample_pos = np.array([x, y]) + kernel[:2] * occlusion_radius
                nx, ny = int(sample_pos[0]), int(sample_pos[1])

                if 0 <= ny < height and 0 <= nx < width:
                    depth_neighbor = depth_map[ny, nx]
                    if abs(depth_neighbor - depth_center) > occlusion_radius:
                        occlusion_count += 1

            ssao_map[y, x] = 1 - occlusion_count / len(kernels)

    return ssao_map


def compute_ssao_with_normals(depth_map, normal_map, kernels, occlusion_radius=15):
    height, width = depth_map.shape
    ssao_map = np.zeros_like(depth_map, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            occlusion_count = 0
            depth_center = depth_map[y, x]
            normal_center = normal_map[y, x]

            for kernel in kernels:
                sample_pos = np.array([x, y]) + kernel[:2] * occlusion_radius
                nx, ny = int(sample_pos[0]), int(sample_pos[1])

                if 0 <= ny < height and 0 <= nx < width:
                    depth_neighbor = depth_map[ny, nx]
                    normal_neighbor = normal_map[ny, nx]

                    # Calculate occlusion based on depth and angle difference
                    if abs(depth_neighbor - depth_center) > occlusion_radius:
                        angle_diff = np.dot(normal_center, normal_neighbor)
                        if angle_diff < 0.9:  # Adjust threshold as needed
                            occlusion_count += 1

            ssao_map[y, x] = 1 - occlusion_count / len(kernels)

    return ssao_map

depth_image = imageio.imread('./000001_000000.depth.png')


#depth_map_denoised = denoise_depth_map(depth_image.astype(np.float32))
#plt.imshow(depth_image)
#plt.show()
#plt.imshow(depth_map_denoised)
#plt.show()

normal_map, normal_map_visual = compute_normal_map(depth_image, smoothing=True )
plt.imshow(normal_map_visual)
plt.show()

rgb_image = imageio.imread('./000001_000000.rgb.png')
kernels = generate_sample_kernels(kernel_size=15)
#ssao_map = compute_ssao_with_random_kernels(depth_image, kernels)


ssao_map = compute_ssao_with_normals(depth_image, normal_map, kernels)
plt.imshow(ssao_map)
plt.show()


#ssao_map = compute_ssao(depth_image)
#plt.imshow(ssao_map)
#plt.show()

ssao_map_normalized = cv2.normalize(ssao_map, None, alpha=00, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
ssao_map_normalized = np.clip(ssao_map_normalized, 0.9, 1)

ssao_applied = rgb_image.copy()
for i in range(3):  # Iterate over the color channels
    ssao_applied[:, :, i] = ssao_applied[:, :, i] * ssao_map_normalized

ssao_applied = np.clip(ssao_applied, 0, 255).astype('uint8')

plt.imshow(rgb_image)
plt.show()
plt.imshow(ssao_applied)
plt.show()
