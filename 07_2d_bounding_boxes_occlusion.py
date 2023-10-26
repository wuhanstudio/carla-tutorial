import carla

import queue
import random

import cv2
import numpy as np

PRELIMINARY_FILTER_DISTANCE = 100

# Part 1

client = carla.Client('localhost', 2000)
world  = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the world spectator
spectator = world.get_spectator() 

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
bp_lib = world.get_blueprint_library()
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# spawn camera
camera_init_trans = carla.Transform(carla.Location(z=2))

# Create a RGB camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Create a depth camera
depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
depth_camera = world.spawn_actor(depth_camera_bp, camera_init_trans, attach_to=vehicle)

def depth_camera_callback(image, depth_image_queue):
    # image.convert(carla.ColorConverter.LogarithmicDepth)
    
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    
    array = array[:, :, :3]     # Take only RGB
    array = array[:, :, ::-1]   # BGR
  
    array = array.astype(np.float32)  # 2ms

    gray_depth = (  (array[:, :, 0] 
                    + array[:, :, 1] * 256.0 
                    + array[:, :, 2] * 256.0 * 256.0) / ((256.0 * 256.0 * 256.0) - 1)
                )  # 2.5ms
    gray_depth = 1000 * gray_depth

    depth_image_queue.put(gray_depth)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

depth_image_queue = queue.Queue()
depth_camera.listen(lambda image: depth_camera_callback(image, depth_image_queue))

# Part 2

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]]).T

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)

    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K   = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

for i in range(20):
    vehicle_bp = bp_lib.filter('vehicle')

    # Exclude bicycle
    car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))

    if npc:
        npc.set_autopilot(True)

# Retrieve all the objects of the level
car_objects = world.get_environment_objects(carla.CityObjectLabel.Car) # doesn't have filter by type yet
truck_objects = world.get_environment_objects(carla.CityObjectLabel.Truck) # doesn't have filter by type yet
bus_objects = world.get_environment_objects(carla.CityObjectLabel.Bus) # doesn't have filter by type yet

env_object_ids = []

for obj in (car_objects + truck_objects + bus_objects):
    env_object_ids.append(obj.id)

# Disable all static vehicles
world.enable_environment_objects(env_object_ids, False) 

edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

def point_is_occluded(point, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    x, y, vertex_depth = map(int, point)
    
    from itertools import product
    neigbours = product((1, -1), repeat=2)

    is_occluded = []
    for dy, dx in neigbours:
        # If the point is on the boundary
        if x == (depth_map.shape[1] - 1) or y == (depth_map.shape[0] - 1):
            is_occluded.append(True)
        # If the depth map says the pixel is closer to the camera than the actual vertex
        elif depth_map[y + dy, x + dx] < vertex_depth:
            is_occluded.append(True)
        else:
            is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def get_vanishing_point(p1, p2, p3, p4):

    k1 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])

    vp_x = (k1 * p3[0] - k2 * p1[0] + p1[1] - p3[1]) / (k1 - k2)
    vp_y = k1 * (vp_x - p3[0]) + p3[1]

    return [vp_x, vp_y]

def clear():
    settings = world.get_settings()
    settings.synchronous_mode = False # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()
    depth_camera.stop()

    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    print("Vehicles Destroyed.")

# Main Loop
vehicle.set_autopilot(True)

edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]


while True:
    try:
        world.tick()

        # Move the spectator to the top of the vehicle 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=50)), carla.Rotation(yaw=-180, pitch=-90)) 
        spectator.set_transform(transform) 

        # Retrieve and reshape the image
        image = image_queue.get()
        depth_map = depth_image_queue.get()

        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < PRELIMINARY_FILTER_DISTANCE:
                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the other vehicle. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA

                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 0:

                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]

                        points_image = []

                        for vert in verts:
                            ray0 = vert - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()

                            if (cam_forward_vec.dot(ray0) > 0):
                                p = get_image_point(vert, K, world_2_camera)
                            else:
                                p = get_image_point(vert, K_b, world_2_camera)

                            points_image.append(p)

                        x_min, x_max = 10000, -10000
                        y_min, y_max = 10000, -10000
                        z_min, z_max = 10000, -10000

                        for edge in edges:
                            p1 = points_image[edge[0]]
                            p2 = points_image[edge[1]]

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            # Both points are out of the canvas
                            if not p1_in_canvas and not p2_in_canvas:
                                continue
                            
                            # Draw 3D Bounding Boxes
                            # cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

                            # Draw 2D Bounding Boxes
                            p1_temp, p2_temp = (p1.copy(), p2.copy())

                            # One of the point is out of the canvas
                            if not (p1_in_canvas and p2_in_canvas):
                                p = [0, 0]

                                # Find the intersection of the edge with the window border
                                p_in_canvas, p_not_in_canvas = (p1, p2) if p1_in_canvas else (p2, p1)
                                k = (p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])

                                x = np.clip(p_not_in_canvas[0], 0, image.width)
                                y = k * (x - p_in_canvas[0]) + p_in_canvas[1]

                                if y >= image.height:
                                    p[0] = (image.height - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = image.height - 1
                                elif y <= 0:
                                    p[0] = (0 - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = 0
                                else:
                                    p[0] = image.width - 1 if x == image.width else 0
                                    p[1] = y

                                p1_temp, p2_temp = (p, p_in_canvas)

                            # Find the rightmost vertex
                            x_max = p1_temp[0] if p1_temp[0] > x_max else x_max
                            x_max = p2_temp[0] if p2_temp[0] > x_max else x_max

                            # Find the leftmost vertex
                            x_min = p1_temp[0] if p1_temp[0] < x_min else x_min
                            x_min = p2_temp[0] if p2_temp[0] < x_min else x_min

                            # Find the highest vertex
                            y_max = p1_temp[1] if p1_temp[1] > y_max else y_max
                            y_max = p2_temp[1] if p2_temp[1] > y_max else y_max

                            # Find the lowest vertex
                            y_min = p1_temp[1] if p1_temp[1] < y_min else y_min
                            y_min = p2_temp[1] if p2_temp[1] < y_min else y_min

                            # No depth information means the point is on the boundary
                            if len(p1_temp) == 3:
                                z_max = p1_temp[2] if p1_temp[2] > z_max else z_max
                                z_min = p1_temp[2] if p1_temp[2] < z_min else z_min

                            if len(p2_temp) == 3:
                                z_max = p2_temp[2] if p2_temp[2] > z_max else z_max
                                z_min = p2_temp[2] if p2_temp[2] < z_min else z_min

                        # Exclude very small bounding boxes
                        if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                            if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                o1 = point_is_occluded((x_min, y_min, z_min), depth_map)
                                o2 = point_is_occluded((x_min, y_min, z_max), depth_map)
                                o3 = point_is_occluded((x_max, y_max, z_min), depth_map)
                                o4 = point_is_occluded((x_max, y_max, z_max), depth_map)

                                # Not all points are occluded
                                if (o1 + o2 + o3 + o4 < 3):
                                    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                                    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                                    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                                    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

        cv2.imshow('3D Bounding Boxes',img)

        if cv2.waitKey(1) == ord('q'):
            clear()
            break

    except KeyboardInterrupt as e:
        clear()
        break

cv2.destroyAllWindows()
