import os, sys
import numpy as np
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from l5kit.data import ChunkedDataset
from l5kit.data.labels import PERCEPTION_LABELS
from l5kit.configs.config import load_metadata
from l5kit.dataset.select_agents import TH_YAW_DEGREE, TH_EXTENT_RATIO, TH_DISTANCE_AV
from l5kit.dataset.select_agents import in_angular_distance, in_extent_ratio

from l5kit.data.map_api import MapAPI, InterpolationMethod
from l5kit.rasterization.semantic_rasterizer import INTERPOLATION_POINTS, cv2_subpixel, CV2_SUB_VALUES
from l5kit.rasterization.rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH
import cv2

# 8.44GB 8.2GB 2.86MB
#
# conda create -n lyft
# conda activate lyft
# conda install python=3.9
# pip install l5kit==1.5.0
# pip install opencv-python
#
# mkdir dataset
# mkdir dataset/train
# mkdir dataset/validate
# mkdir dataset/semantic_map
# wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/train.tar
# wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/validate.tar
# wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar

# tar -xf train.tar -C dataset/train
# tar -xf validate.tar -C dataset/validate
# tar -xf semantic_map.tar -C dataset/semantic_map
# python lyft.py dataset 5FPS --frameskip 2 --fraction 4


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=str)
parser.add_argument("target_folder", type=str)
parser.add_argument("zarr_folders", nargs="*", default=["validate", "train"])
parser.add_argument("map_folder", nargs="?", type=str, default="semantic_map")
parser.add_argument("--frameskip", type=int, default=2)
parser.add_argument("--map_scale", type=int, default=1)
parser.add_argument("--split", type=int, default=4)
parser.add_argument("--workers", type=int, default=None)
settings = parser.parse_args()

FRAMESKIP = settings.frameskip
MAP_SCALE = settings.map_scale
MAP_FILE_NAME = "overall-map"
WORKERS = min(30, multiprocessing.cpu_count()) if settings.workers is None else settings.workers
MAP_PATH = os.path.join(settings.data_root, settings.map_folder, "semantic_map.pb")
DATASET_META = os.path.join(settings.data_root, settings.map_folder, "meta.json")

EGO_EXTENT = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
def generate(dataset, host, start_frame, end_frame, path):
    name = "{}-{}-{}".format(host, start_frame, end_frame)

    agents = dict()
    tracks = dict()
    count = 0

    frames = dataset.frames[start_frame:end_frame]
    agents_start_index = frames[0]["agent_index_interval"][0]
    agents_end_index = frames[-1]["agent_index_interval"][1] 
    frame_agents = iter(dataset.agents[agents_start_index:agents_end_index])
    n_agents = frames["agent_index_interval"][:,1] - frames["agent_index_interval"][:,0]

    for fid, frame in enumerate(frames):
        ego_x = frame["ego_translation"][0]
        ego_y = frame["ego_translation"][1]
        need_record = (fid-start_frame) % FRAMESKIP == 0
        if need_record:
            R = frame["ego_rotation"]
            singular = (R[0][0]*R[0][0] + R[1][1]*R[1][1])**0.5 < 1e-6
            heading = 0 if singular else np.arctan2(R[1][0], R[0][0]) # yaw
            agent_id = 0
            agent_type = "EGO"
            if agent_id not in agents: agents[agent_id] = []
            agents[agent_id].append((ego_x, ego_y, heading, agent_type, EGO_EXTENT, 0, fid))
        for _ in range(n_agents[fid]):
            agent = next(frame_agents)
            prob = agent["label_probabilities"]
            agent_type = np.argmax(prob)
            agent_type = "_".join(PERCEPTION_LABELS[agent_type].split("_")[2:])
            if agent_type in ["CAR", "BUS", "VAN", "TRUCK", "EMERGENCY_VEHICLE", "OTHER_VEHICLE"]:
                agent_type = "VEHICLE"
            elif agent_type not in [
                "TRAM",
                "PEDESTRIAN", 
                "ANIMAL",
                "BICYCLE", "MOTORCYCLE", "CYCLIST", "MOTORCYCLIST"
            ]:
                continue
            x = agent["centroid"][0]
            y = agent["centroid"][1]
            heading = agent["yaw"]
            extent = agent["extent"]
            track_id = agent["track_id"]

            # filter vehicles for prediction in the same way with lyft dataset
            # https://github.com/woven-planet/l5kit/blob/71eb4dae2c8230f7ca2c60e5b95473c91c715bb8/l5kit/l5kit/dataset/select_agents.py#L76
            tracked = track_id in tracks and tracks[track_id] is not None 
            lost = False
            if max(prob) < 0.5:
                agent_type += "_IGNORE"
                lost = True
            elif np.linalg.norm([x-ego_x, y-ego_y]) > TH_DISTANCE_AV:
                agent_type += "_IGNORE"
                lost = True
            if not tracked or fid - tracks[track_id][1][0] != 1 or \
            not in_angular_distance(tracks[track_id][1][1], heading, TH_YAW_DEGREE) or \
            not in_extent_ratio(tracks[track_id][1][2], extent, TH_EXTENT_RATIO):
                # lose track, consider the current agent as a new agent
                agent_id = None
            else:
                agent_id = tracks[track_id][0]
            if need_record:
                if agent_id is None:
                    count += 1
                    agent_id = count
                if agent_id not in agents:
                    agents[agent_id] = []
                agents[agent_id].append((x, y, heading, agent_type, extent, track_id, fid))
            if lost:
                tracks[track_id] = None
            else:
                tracks[track_id] = agent_id, (fid, heading, extent)
    content = []
    x, y = [], []
    for agent_id, record in agents.items():
        if len(record) < 2: continue
        for x_, y_, yaw, agent_type, agent_extent, track_id, fid in record:
            content.append("{:<3d} {:<4d} {:10.4f} {:10.4f} {:7.4f} {} {:.4f}/{:.4f}/{:.4f} {:d}".format(
                fid, agent_id, x_, y_, yaw, agent_type, agent_extent[0], agent_extent[1], agent_extent[2], track_id
            ))
            x.append(x_)
            y.append(y_)
    if not content: return None

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    with open(os.path.join(path, "{}.txt".format(name)), "w") as f:
        f.write("\n".join(content))
    return x_min, x_max, y_min, y_max

if __name__ == "__main__":
    print("Loading dataset...")
    dataset_meta = load_metadata(DATASET_META)
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    map_path = os.path.join(settings.target_folder, "map")

    args = []
    subfolders = []
    dataset = {}
    for zarr in settings.zarr_folders:
        if os.path.exists(os.path.join(settings.data_root, zarr)):
            basename = zarr
            zarr = os.path.join(settings.data_root, zarr)
            if not zarr.endswith(".zarr") and os.path.exists(os.path.join(zarr, "{}.zarr".format(basename))):
                zarr = os.path.join(zarr, "{}.zarr".format(basename))
        dataset[basename] = ChunkedDataset(zarr).open()
        scenes = dataset[basename].scenes
        n_scenes = len(scenes)
        n = n_scenes // settings.split
        rem = n_scenes % settings.split
        e, batch = 0, 0
        target_folder = settings.target_folder
        for i, scene in enumerate(scenes):
            if i == e: 
                s = e
                e = s + n + (1 if batch < rem else 0)
                batch += 1
            start_frame = scene["frame_index_interval"][0]
            end_frame   = scene["frame_index_interval"][1]
            host = scene["host"]
            path = os.path.join(settings.target_folder, os.path.splitext(os.path.basename(zarr))[0])
            if settings.split > 1:
                path = os.path.join(path, str(batch))
            if path not in subfolders: subfolders.append(path)
            args.append((dataset[basename], host, start_frame, end_frame, path))

    os.makedirs(map_path, exist_ok=True)
    for p in subfolders: os.makedirs(p, exist_ok=True)

    done = 0
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=WORKERS) as p:
        futures = [p.submit(generate, *arg) for arg in args]
        for fut in as_completed(futures):
            done += 1
            sys.stderr.write("\r{}/{} {:.2%}".format(done, len(args), done/len(args)))
    corner = None
    for fut in futures:
        d = fut.result()
        if d is None: continue
        if corner is None:
            corner = d
        else:
            corner = (min(d[0], corner[0]), max(d[1], corner[1]), min(d[2], corner[2]), max(d[3], corner[3]))
    x_min, x_max, y_min, y_max = corner

    print()
    print("Valid map range: ", corner)
    print("Generating map...")
    print()

    x_min = x_min - 300
    x_max = x_max + 300
    y_min = y_min - 300
    y_max = y_max + 300

    x_min = int(np.floor(x_min))
    x_max = int(np.ceil(x_max))
    y_min = int(np.floor(y_min))
    y_max = int(np.ceil(y_max))

    x_center = (x_min+x_max)*0.5
    y_center = (y_min+y_max)*0.5
    canvas_size = int(np.ceil((y_max-y_min)*MAP_SCALE)), int(np.ceil((x_max-x_min)*MAP_SCALE))
    height = 1.*canvas_size[0]/MAP_SCALE
    width = 1.*canvas_size[1]/MAP_SCALE

    def indices_in_bounds(bounds):
        x_min_in = x_max > bounds[:, 0, 0]
        y_min_in = y_max > bounds[:, 0, 1]
        x_max_in = x_min < bounds[:, 1, 0]
        y_max_in = y_min < bounds[:, 1, 1]
        return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]

    map_api = MapAPI(MAP_PATH, world_to_ecef)

    lanes = []
    lanes_mask = [[], []]
    road_dividers = [
        "[10]", "[11]", "[12]", # CURB_RED, CURB_YELLOW, CURB
        "[6]", # DOUBLE_YELLOW_SOLID
        # "[8]", # DOUBLE_YELLOW_SOLID_FAR_DASHED_NEAR
        # "[9]", # DOUBLE_YELLOW_DASHED_FAR_SOLID_NEAR
    ] 
    for idx in indices_in_bounds(map_api.bounds_info["lanes"]["bounds"]):
        elem_id = map_api.bounds_info["lanes"]["ids"][idx]
        coords = map_api.get_lane_as_interpolation(
            elem_id, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
        )
        lanes.append(coords["xyz_left"][:, :2])
        lanes.append(coords["xyz_right"][::-1, :2])

        if map_api[elem_id].element.lane.left_boundary.divider_type.__repr__() in road_dividers:
            lanes_mask[0].append(True)
            lanes_mask[1].append(False)
        else:
            lanes_mask[0].append(False)
            lanes_mask[1].append(True)
        if map_api[elem_id].element.lane.right_boundary.divider_type.__repr__() in road_dividers:
            lanes_mask[0].append(True)
            lanes_mask[1].append(False)
        else:
            lanes_mask[0].append(False)
            lanes_mask[1].append(True)

    crosswalks = []
    for idx in indices_in_bounds(map_api.bounds_info["crosswalks"]["bounds"]):
        elem_id = map_api.bounds_info["crosswalks"]["ids"][idx]
        coords = map_api.get_crosswalk_coords(elem_id)
        crosswalks.append(coords["xyz"][:,:2])

    H = np.array([
        [0., -MAP_SCALE, (y_center+0.5*height)*MAP_SCALE],
        [MAP_SCALE, 0., -(x_center-0.5*width)*MAP_SCALE],
        [0., 0., 1.]
    ]) # counts from left-top corner


    im_lane = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    im_crosswalk = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    im_lane_divider = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    im_road_divider = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    if lanes:
        lanes = np.reshape(lanes, (-1, 2, 1))
        lanes = np.concatenate((lanes, np.ones((lanes.shape[0], 1, 1))), 1)
        lanes = (H @ lanes).squeeze(-1)[:,:2]
        # cv2.fillPoly function takes points in the format of (column, row)
        lanes = lanes[:,::-1] # convert from row, column to column, row
        lanes = cv2_subpixel(lanes)
        for lane_area in lanes.reshape((-1,INTERPOLATION_POINTS*2, 2)):
            cv2.fillPoly(im_lane, [lane_area], 255, **CV2_SUB_VALUES)
        lanes = lanes.reshape((-1,INTERPOLATION_POINTS, 2))
        cv2.polylines(im_road_divider, lanes[lanes_mask[0]], False, 255, **CV2_SUB_VALUES)
        cv2.polylines(im_lane_divider, lanes[lanes_mask[1]], False, 255, **CV2_SUB_VALUES)

    if crosswalks:
        indices = np.cumsum([len(_) for _ in crosswalks])
        crosswalks = np.concatenate(crosswalks).reshape(-1, 2, 1)
        crosswalks = np.concatenate((crosswalks, np.ones((crosswalks.shape[0], 1, 1))), 1)
        crosswalks = (H @ crosswalks).squeeze(-1)[:,:2]
        # cv2.fillPoly function takes points in the format of (column, row)
        crosswalks = crosswalks[:,::-1] # convert from row, column to column, row
        crosswalks = cv2_subpixel(crosswalks)
        crosswalks = np.split(crosswalks, indices[:-1])
        cv2.fillPoly(im_crosswalk, crosswalks, 255, **CV2_SUB_VALUES)

    im_lane = (im_lane/255).astype(np.float32)*0.75 + (im_crosswalk/255).astype(np.float32)*0.25
    im_road_divider = im_road_divider.astype(np.float32)/255
    im_lane_divider = im_lane_divider.astype(np.float32)/255
    semantic_map = np.stack((im_lane, im_road_divider, im_lane_divider))
    semantic_map = (semantic_map*2-1).clip(-1, 1)

    with open(os.path.join(map_path, "{}.pkl".format(MAP_FILE_NAME)), "wb") as f:
        pickle.dump((semantic_map, H), f)
