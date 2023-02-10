from typing import Optional, List, Sequence

import os, sys
import torch
import numpy as np
import io, pickle

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

class Dataloader(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None #iter(self.sampler)
        def __iter__(self):
            # same with BatchSampler, but StopIteration every n batches
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None: 
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    def __init__(self,
        files: List[str], ob_horizon: int, pred_horizon: int,
        batch_size: int, drop_last: bool=False, shuffle: bool=False, batches_per_epoch=None, 
        frameskip: int=1, inclusive_groups: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        flip: bool=False, rotate: bool=False, scale: bool=False,

        min_ob_horizon: Optional[int]=None, traj_max_overlap: Optional[int]=None,
        ob_radius: Optional[int]=None,
        map_dir: Optional[str]=None, map_size: int=224, map_scale: int=1, preload_map: bool=True
    ):
        super().__init__()
        self.min_ob_horizon = ob_horizon if min_ob_horizon is None else min_ob_horizon
        self.ob_horizon = ob_horizon
        if self.ob_horizon < self.min_ob_horizon:
            self.min_ob_horizon, self.ob_horizon = self.ob_horizon, self.min_ob_horizon
        self.pred_horizon = pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.use_map = map_dir
        self.map_scale = map_scale
        self.map_size = map_size
        self.flip = flip and not self.use_map
        self.rotate = rotate and not self.use_map
        self.scale = scale and not self.use_map

        self.device = device
        self.traj_max_overlap = traj_max_overlap
        self.ob_radius = ob_radius
        self.preload_map = preload_map

        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        else:
            # assert(len(inclusive_groups) == len(files))
            inclusive_groups = [inclusive_groups for _ in range(len(files))]

        print(" Scanning files...")
        files_ = []
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend([(os.path.join(root, f), incl_g) \
                    for root, _, fs in os.walk(path) \
                    for f in fs if f.endswith(".txt")])
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])
        assert len(data_files) > 0, "No valid files found from {}".format(files)

        data = []
        self.map = dict()
        if self.use_map:
            if os.path.isdir(self.use_map):
                map_files = [os.path.join(root, f) for root, _, files in os.walk(self.use_map) for f in files]
            else:
                map_files = [self.use_map]
            for i, map_file in enumerate(map_files, 1):
                map_name = os.path.splitext(os.path.basename(map_file))[0]
                self.map[map_name] = map_file #path
            self.use_map = True
            print(" {} map file{} founded.".format(len(self.map), "s" if len(self.map) > 1 else ""))
        sys.stdout.write("\r\033[K Loading data files...{}/{}".format(
            0, len(data_files)
        ))
        
        
        done = 0
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
            futures = [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]
            for fut in as_completed(futures):
                done += 1
                sys.stdout.write("\r\033[K Loading data files...{}/{}".format(
                    done, len(data_files)
                ))
            for fut in futures:
                item = fut.result()
                if item is not None:
                    data.extend(item)
                sys.stdout.write("\r\033[K Loading data files...{}/{} ".format(
                    done, len(data_files)
                ))
        # disable augmentation if heading angle is specified
        if data[-1][3][1] is not None:
            self.flip = False
            self.rotate = False
            self.scale = False
        self.data = np.array(data, dtype=object)
        del data
        print("\n   {} trajectories loaded.".format(len(self.data)))
        
        if self.use_map and self.preload_map:
            MAP_SIZE = self.map_size*self.map_scale
            TOP = MAP_SIZE//2
            LEFT = MAP_SIZE//4
            self.EXT = int(np.ceil(MAP_SIZE//4*((9+4)**0.5)))

            BOTTOM = MAP_SIZE - TOP
            RIGHT = MAP_SIZE - LEFT

            self.TOP = self.EXT-TOP
            self.BOTTOM = self.EXT+BOTTOM
            self.LEFT = self.EXT-LEFT
            self.RIGHT = self.EXT+RIGHT
        
            map_files = set()
            for item in self.data:
                map_name = item[3][0]
                map_files.add(map_name)
            compressed = len(map_files) > 6000
            if compressed: self.preload_map = False
            if len(map_files) > 10:
                with ThreadPoolExecutor(max_workers=min(len(map_files), torch.get_num_threads(), 20)) as p:
                    futs = [p.submit(self.__class__.load_map, self.map[map_name], compressed=compressed)
                        for map_name in map_files]
                    done = 0
                    for fut in as_completed(futs):
                        done += 1
                        sys.stdout.write("\r\033[K Loading map files...{}/{}".format(
                            done, len(map_files)
                        ))
                    for map_name, fut in zip(map_files, futs):
                        self.map[map_name] = fut.result()
            else:
                for i, map_name in enumerate(map_files, 1):
                    if type(self.map[map_name]) == str:
                        self.map[map_name] = self.load_map(self.map[map_name], compressed=compressed)
                    sys.stdout.write("\r\033[K Loading map files...{}/{}".format(
                        i, len(map_files)
                    ))
            print()

        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch


    def collate_fn(self, batch):
        X, Y, NEIGHBOR = [], [], []
        R, M = [], []
        L, seq_len = [], None
        for item in batch:
            hist, future, neighbor = item[0], item[1], item[2]
            seq_len = item[-1]

            if self.use_map:
                semantic_map = item[3]
                rot = item[4]
                R.append(rot)
                M.append(semantic_map)
            else:
                hist_shape = hist.shape
                neighbor_shape = neighbor.shape
                hist = np.reshape(hist, (-1, 2))
                neighbor = np.reshape(neighbor, (-1, 2))
                if self.flip:
                    if self.rng.randint(2):
                        hist[..., 1] *= -1
                        future[..., 1] *= -1
                        neighbor[..., 1] *= -1
                    if self.rng.randint(2):
                        hist[..., 0] *= -1
                        future[..., 0] *= -1
                        neighbor[..., 0] *= -1
                if self.rotate:
                    rot = self.rng.random() * (np.pi+np.pi) 
                    s, c = np.sin(rot), np.cos(rot)
                    r = np.asarray([
                        [c, -s],
                        [s,  c]
                    ])
                    hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)
                    future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                    neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
                if self.scale:
                    s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                    hist = s * hist
                    future = s * future
                    neighbor = s * neighbor
                hist = np.reshape(hist, hist_shape)
                neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
            L.append(seq_len)

        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        ret = [x, y, neighbor]
        if self.use_map:
            with torch.no_grad():
                r = torch.stack(R, 0).to(self.device)
                m = torch.stack(M, 0).to(self.device)
                grid = torch.nn.functional.affine_grid(r, m.size(), align_corners=False)
                m = torch.nn.functional.grid_sample(m, grid, align_corners=False)
                m = m[..., self.TOP:self.BOTTOM, self.LEFT:self.RIGHT]
                m = m.unsqueeze_(0)
            ret.append(m)
        if seq_len is not None:
            seq_len = torch.as_tensor(L, dtype=torch.long, device=self.device)
            ret.append(seq_len)
        return ret

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        hist, future, neighbor = item[0], item[1], item[2]
        seq_len = item[-1]
        if self.use_map:
            map_name, angle = item[3]
            if self.preload_map:
                semantic_map, H = self.map[map_name]
            else:
                buf = self.map[map_name] #self.load_map(self.map[map_name])
                if type(buf) == str:
                    semantic_map, H = self.load_map(buf, compressed=False)
                else: 
                    buf.seek(0)
                    loaded = np.load(buf)
                    semantic_map = torch.tensor(loaded["M"], dtype=torch.float32)
                    H = loaded["H"]
            x_center, y_center = hist[0][0], hist[0][1]
            r, c, _ = H @ [x_center, y_center, 1]
            r = int(np.floor(r))
            c = int(np.floor(c))
            sin, cos = np.sin(angle), np.cos(angle)
            rot = torch.tensor([
                [cos, -sin, 0],
                [sin,  cos, 0],
            ], dtype=torch.float32)
            semantic_map = semantic_map[:, r-self.EXT:r+self.EXT, c-self.EXT:c+self.EXT]#.clone().detach()
            return hist, future, neighbor, semantic_map, rot, seq_len
        return hist, future, neighbor, seq_len

    @staticmethod
    def load(self, filename, inclusive_groups):
        if os.path.isdir(filename): return None

        min_horizon = (self.min_ob_horizon-1+self.pred_horizon)*self.frameskip
        with open(filename, "r") as record:
            data = self.load_traj(record)
        data = self.extend(data, self.frameskip)
        time = np.sort(list(data.keys()))
        if len(time) < min_horizon+1: return None

        # extend the observation radius a little bit to prevent computation errors
        ob_radius = None if self.ob_radius is None else self.ob_radius + 0.5 

        info_file = filename.replace(".txt", ".info")
        if os.path.exists(info_file):
            ts, maps = [], []
            with open(info_file, "r") as map_file:
                for line in map_file.readlines():
                    if self.use_map:
                        t, map_name = line.split()
                        maps.append(map_name)
                    else:
                        t = line.split()[0]
                    ts.append(np.where(time == int(t))[0][0])
            segments = [(t, ts[i] if i < len(ts) else len(time)) for i, t in enumerate(ts, 1)]
        else:
            segments = [(0, len(time))]
            if self.use_map:
                map_name = os.path.splitext(os.path.basename(filename))[0]
                if map_name not in self.map:
                    assert(len(self.map) == 1)
                    map_name = next(iter(self.map.keys()))
                maps = [map_name]
        items = []

        ob_gap = self.ob_horizon - self.min_ob_horizon
        for seg, (s, e) in enumerate(segments):
            tid_curr = s + (self.min_ob_horizon-1)*self.frameskip
            tid_final = e - self.pred_horizon*self.frameskip
            timestamp = dict()
            while tid_curr < tid_final:
                tid_start = tid_curr - (self.ob_horizon-1)*self.frameskip
                tid_end = tid_curr + self.pred_horizon*self.frameskip
                idx_ego = []
                idx_all = []
                first_frame = dict()
                for i, tid in enumerate(range(tid_start, tid_end+1, self.frameskip)):
                    if tid < s: continue
                    t = time[tid]
                    idx_curr = [aid for aid, d in data[t].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]
                    if not idx_curr: # interrupted by empty frame
                        idx_ego = []
                        if i >= ob_gap: break
                    idx_ego = np.intersect1d(idx_ego, idx_curr) # remove those not appear at current frame
                    if i <= ob_gap:
                        for idx in idx_curr:
                            if idx not in idx_ego:
                                first_frame[idx] = (i, tid)
                        idx_ego = np.union1d(idx_ego, idx_curr) # add to ego agent list if it appears at available observation phase
                    if i >= ob_gap and len(idx_ego) == 0:
                        break
                    idx_all.extend(data[t].keys())
                if self.min_ob_horizon != self.ob_horizon:
                    tid_next = tid_end+self.frameskip  # remove those whose length is shorter than the observation windows but whose actually trajectory length is longer 
                    if tid_next < e and tid_next < len(time) and time[tid_next] in data:
                        removed = []
                        for idx in idx_ego:
                            if first_frame[idx][0] == 0: continue
                            t = time[tid_next]
                            if idx in data[t] and any(g in inclusive_groups for g in data[t][idx][-1]):
                                removed.append(idx)
                        idx_ego = np.setdiff1d(idx_ego, removed)
                if self.traj_max_overlap is not None and self.traj_max_overlap < self.ob_horizon+self.pred_horizon:
                    overlapped = []
                    for idx in idx_ego:
                        if idx in timestamp and first_frame[idx][1] < timestamp[idx]:
                            overlapped.append(idx)
                        else:
                            timestamp[idx] = tid_end - (self.traj_max_overlap-1)*self.frameskip
                    idx_ego = np.setdiff1d(idx_ego, overlapped)
                if len(idx_ego):
                    data_dim = 6 # x, y, vx, vy, ax, ay
                    heading_dim_index = 6
                    neighbor_idx = np.setdiff1d(idx_all, idx_ego)
                    if len(idx_ego) == 1 and len(neighbor_idx) == 0:
                        idx = idx_ego[0]
                        agents = np.array([
                            [data[time[tid]][idx][:data_dim] if time[tid] in data and idx in data[time[tid]] and tid >= 0 else [1e9]*data_dim] + [[1e9]*data_dim]
                            for tid in range(tid_start, tid_end+1, self.frameskip)
                        ]) # L x 2 x 6
                    else:
                        agents = np.array([
                            [data[time[tid]][i][:data_dim] if time[tid] in data and i in data[time[tid]] and tid >= 0 else [1e9]*data_dim for i in idx_ego] +
                            [data[time[tid]][j][:data_dim] if time[tid] in data and j in data[time[tid]] else [1e9]*data_dim for j in neighbor_idx]
                            for tid in range(tid_start, tid_end+1, self.frameskip)
                        ])  # L X N x 6
                    for i, idx in enumerate(idx_ego):
                        heading = data[time[first_frame[idx][1]]][idx][heading_dim_index]
                        hist = agents[:self.ob_horizon,i]  # L_ob x 6
                        future = agents[self.ob_horizon:,i,:2]  # L_pred x 2
                        neighbor = agents[:, [d for d in range(agents.shape[1]) if d != i]] # L x (N-1) x 6

                        if ob_radius is not None:
                            dp = neighbor[:,:,:2] - agents[:,i:i+1,:2]
                            dist = np.linalg.norm(dp, axis=-1) # L x (N-1)
                            valid = dist <= ob_radius  # L x (N-1)
                            valid = np.any(valid, axis=0) # N-1
                            neighbor = neighbor[:, valid]

                        if first_frame[idx][0] > 0:
                            # pad the observation if the observation horizon is less than the maximal horizon setting
                            seq_len = self.ob_horizon - first_frame[idx][0]
                            padded_hist = np.zeros_like(hist)
                            padded_hist[:seq_len] = hist[-seq_len:]
                            hist = padded_hist
                            padded_neighbor = np.zeros_like(neighbor)
                            padded_neighbor[self.ob_horizon:] = neighbor[self.ob_horizon:]
                            padded_neighbor[:seq_len] = neighbor[self.ob_horizon-seq_len:self.ob_horizon]
                            neighbor = padded_neighbor
                        else:
                            seq_len = self.ob_horizon
                        
                        if self.use_map:
                            m = maps[seg]
                            items.append((hist, future, neighbor, (m, -heading), seq_len if self.min_ob_horizon != self.ob_horizon else None))
                        else:
                            items.append((hist, future, neighbor, (None, None if heading is None else -heading), seq_len if self.min_ob_horizon != self.ob_horizon else None))
                tid_curr += 1
        traj = items
        items = []

        for hist, future, neighbor, (map_name, angle), seq_len in traj:
            if angle is not None:
                # localize the trajectory according to heading angle
                x0 = hist[0][0]
                y0 = hist[0][1]
                s, c = np.sin(angle), np.cos(angle)
                R = np.asarray([
                    [c, -s],
                    [s,  c],
                ])

                hist[...,:2] -= [x0, y0]
                future -= [x0, y0]
                neighbor[...,:2] -= [x0, y0]
                hist = (R @ np.reshape(hist, (-1,2,1))).reshape(*hist.shape)
                future = (R @ np.reshape(future, (-1,2,1))).reshape(*future.shape)
                neighbor = (R @ np.reshape(neighbor, (-1,2,1))).reshape(*neighbor.shape)
                hist[...,:2] += [x0, y0]
                future += [x0, y0]
                neighbor[...,:2] += [x0, y0]

            hist = np.float32(hist)
            future = np.float32(future)
            neighbor = np.float32(neighbor)
            items.append((hist, future, neighbor, (map_name, angle), seq_len))
        return items

    @staticmethod
    def load_map(map_file, compressed=False):
        with open(map_file, "rb") as f:
            semantic_map, H = pickle.load(f)
        if compressed:
            buf = io.BytesIO()
            np.savez_compressed(buf, M=semantic_map, H=H)
            return buf
        semantic_map = torch.tensor(semantic_map, dtype=torch.float32)
        return semantic_map, H

    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])
        dt = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts)
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                time = np.insert(time, i+1, time[i]+dt)
            i += 1
        # ignore those only appearing at one frame
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and \
                (t1 is None or t1 not in data or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        # extend v
        for tid in range(len(time)-frameskip):
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid-1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        # extend a
        for tid in range(len(time)-frameskip):
            t_1 = None if tid < frameskip else time[tid-frameskip]
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                if t_1 is None or i not in data[t_1]:
                    # first appearing frame, pick value from the next frame
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
        return data

    def load_traj(self, file):
        data = {}
        for row in file.readlines():
            item = row.split()
            if not item: continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            if len(item) > 5:
                heading = float(item[4])
                group = item[5].split("/")
            elif len(item) > 4:
                heading = None
                group = item[4].split("/")
            else:
                heading = None
                group = None
            if t not in data:
                data[t] = {}
            data[t][idx] = [x, y, heading, group]
        return data
