import os, sys, time, gc
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from context_vae import ContextVAE
from data import Dataloader
from utils import ADE_FDE, seed, clustering, get_rng_state, set_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--map_dir", type=str, default=None)
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--test_map_dir", type=str, default=None)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--master_addr", type=str, default="localhost")
parser.add_argument("--master_port", type=str, default="29500")

if __name__ == "__main__":
    settings = parser.parse_args()
    if not settings.test_map_dir: settings.test_map_dir = settings.map_dir

    import importlib
    print(os.path.dirname(settings.config))
    spec = importlib.util.spec_from_file_location("config", settings.config, 
        submodule_search_locations=[os.path.dirname(settings.config)])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device
    device = torch.device(device)
    settings.workers = max(1, settings.workers)
    assert(settings.rank is not None or settings.workers <= 1)

    seed(settings.seed)
    init_rng_state = get_rng_state(device)
    rng_state = init_rng_state
    if settings.rank is not None:
        os.environ["MASTER_ADDR"] = settings.master_addr
        os.environ["MASTER_PORT"] = settings.master_port
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.set_num_threads(min(torch.get_num_threads()//device_count, 20))

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    preload_kwargs = dict(
        num_workers=6, pin_memory=False, prefetch_factor=2, persistent_workers=True,
    ) if config.preload_data else dict()
    kwargs = dict(
        batch_first=False,
        device="cpu" if config.preload_data else device,
        seed=settings.seed)
    train_data, test_data = None, None

    if settings.test:
        print(settings.test)
        test_dataset = Dataloader(
            settings.test, **kwargs, 
            **config.test_dataloader,
            map_dir=settings.test_map_dir,
            shuffle=False
        )
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler, 
            **preload_kwargs
        )

    if settings.train:
        print(settings.train)
        config.train_dataloader["batch_size"] //= settings.workers
        train_dataset = Dataloader(
            settings.train, **kwargs, 
            **config.train_dataloader,
            map_dir=settings.map_dir,
            shuffle=True
        )
        train_data = torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_dataset.batch_sampler, 
            **preload_kwargs
            )
        batches = train_dataset.batches_per_epoch

    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = ContextVAE(**config.model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start_epoch = 0
    if settings.ckpt:
        ckpt = None
        if os.path.isfile(settings.ckpt):
            ckpt = settings.ckpt
            settings.ckpt = os.path.dirname(ckpt)
        if ckpt is not None and not settings.train:
            ckpt_best = ckpt
        else:
            ckpt_best = os.path.join(settings.ckpt_dir, "ckpt-best")
        if ckpt is None:
            ckpt = os.path.join(settings.ckpt, "ckpt-last")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            ade_d_best = state_dict["ade_d"]
            fde_d_best = state_dict["fde_d"]
        else:
            ade_best = 100000
            fde_best = 100000
            ade_d_best = 100000
            fde_d_best = 100000
        if not settings.train:
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
            rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            print("Epoch:", state_dict["epoch"])
            print("eval: {:.4f}/{:.4f}, {:.4f}/{:.4f}".format(state_dict["ade"], state_dict["ade_d"], state_dict["fde"], state_dict["fde_d"]))
            start_epoch = state_dict["epoch"]
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.epochs else config.epochs

    if settings.rank is not None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=settings.rank, world_size=settings.workers)
        device_ids = None if settings.device is None else [settings.device]
        if model.use_map:
            # resnet uses batch norm
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=device_ids)
        model.state_dict = model.module.state_dict
        model.loss = model.module.loss
        is_chief = settings.rank == 0
    else:
        is_chief = True

    if is_chief and settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None

    if train_data is not None:
        log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
        progress = 20/batches if batches > 20 else 1
        optimizer.zero_grad()

    for epoch in range(start_epoch+1, end_epoch+1):
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        if train_data is not None and epoch <= config.epochs:
            print("Epoch {}/{}".format(epoch, config.epochs))
            tic = time.time()
            set_rng_state(rng_state, device)
            losses = {}
            model.train()
            sys.stdout.write(log_str.format(
                cur_batch=0, done="", remain="."*int(batches*progress),
                time=round(time.time()-tic), comment=""))
            for batch, item in enumerate(train_data):
                if train_dataset.device != device:
                    item = [_.to(device) for _ in item]
                res = model(*item)
                loss = model.loss(*res)
                loss["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                for k, v in loss.items():
                    if k not in losses: 
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k]*batch+v.item())/(batch+1)
                sys.stdout.write(log_str.format(
                    cur_batch=batch+1, done="="*int((batch+1)*progress),
                    remain="."*(int(batches*progress)-int((batch+1)*progress)),
                    time=round(time.time()-tic),
                    comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()]) + \
                        " - lr: {:e}".format(optimizer.param_groups[0]["lr"])
                ))
            rng_state = get_rng_state(device)
            print()

        gc.collect()
        torch.cuda.empty_cache()
        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        ade, fde, ade_d, fde_d = 10000, 10000, 10000, 10000
        perform_test = (train_data is None or config.test_since <= epoch) and test_data is not None
        if perform_test:
            sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                0, len(test_dataset)
            ))
            tic = time.time()
            model.eval()

            ADE, FDE = [], []
            ADE_d, FDE_d = [], []
            m = None # stub for map tensor data
            set_rng_state(init_rng_state, device)
            batch = 0
            with torch.no_grad():
                for item in test_data:
                    if test_dataset.device != device:
                        item = [_.to(device) for _ in item]
                    x, y, neighbor, *m = item
                    batch += x.size(1)
                    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                        batch, len(test_dataset)
                    ))

                    tic = time.time()
                    if config.clustering:
                        if test_dataset.use_map:
                            y_ = model(x, neighbor, *m, n_predictions=config.clustering)
                        else:
                            y_ = [model(x, neighbor, *m, n_predictions=config.pred_samples) for _ in range(int(np.ceil(config.fpc/config.pred_samples)))]
                            y_ = torch.cat(y_, 0)
                        # y_: n_samples x PRED_HORIZON x N x 2
                        cand = []
                        for i in range(y_.size(-2)):
                            traj, counts = clustering(y_[..., i, :].cpu().numpy(), n_samples=config.pred_samples)
                            cand.append(traj)
                        y_ = torch.as_tensor(np.stack(cand, 2), device=y_.device, dtype=y_.dtype)
                    else:
                        y_ = model(x, neighbor, *m, n_predictions=config.pred_samples) # n_samples x PRED_HORIZON x N x 2
                    ade, fde = ADE_FDE(y_, y)
                    ade = torch.min(ade, dim=0)[0]
                    fde = torch.min(fde, dim=0)[0]
                    ADE.append(ade)
                    FDE.append(fde)

                    y_ = model(x, neighbor, *m, n_predictions=0)
                    ade, fde = ADE_FDE(y_, y)
                    ADE_d.append(ade)
                    FDE_d.append(fde)

                ADE_d = torch.cat(ADE_d)
                FDE_d = torch.cat(FDE_d)
                ade_d = ADE_d.mean()
                fde_d = FDE_d.mean()

                ADE = torch.cat(ADE)
                FDE = torch.cat(FDE)

                if type(model) == DDP:
                    ade = ADE.sum()
                    fde = FDE.sum()
                    ade_d = ADE_d.sum()
                    fde_d = FDE_d.sum()
                    n = torch.tensor(ADE.size(0), dtype=torch.int64, device=ade.device)
                    dist.all_reduce(ade, dist.ReduceOp.SUM)
                    dist.all_reduce(fde, dist.ReduceOp.SUM)
                    dist.all_reduce(ade_d, dist.ReduceOp.SUM)
                    dist.all_reduce(fde_d, dist.ReduceOp.SUM)
                    dist.all_reduce(n, dist.ReduceOp.SUM)
                    ade /= n
                    fde /= n
                    ade_d /= n
                    fde_d /= n
                else:
                    ade = ADE.mean()
                    fde = FDE.mean()
                    ade_d = ADE_d.mean()
                    fde_d = FDE_d.mean()

                ade = ade.item()
                fde = fde.item()
                ade_d = ade_d.item()
                fde_d = fde_d.item()
                sys.stdout.write("\r\033[K ADE: {:.4f}/{:.4f}; FDE: {:.4f}/{:.4f} -- time: {}s".format(ade, ade_d, fde, fde_d, int(time.time()-tic)))
            print()
        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
        if is_chief and losses is not None and settings.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
                if perform_test:
                    logger.add_scalars("eval", dict(
                        ADE_min=ade, FDE_min=fde,
                        ADE_deter=ade_d, FDE_deter=fde_d
                    ), epoch)
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade, fde=fde, ade_d=ade_d, fde_d=fde_d, epoch=epoch, rng_state=rng_state
            )
            torch.save(state, ckpt)
            if ade < ade_best:
                ade_best = ade
                fde_best = fde
                state = dict(
                    model=state["model"],
                    ade=ade, fde=fde, ade_d=ade_d, fde_d=fde_d, epoch=epoch, rng_state=rng_state
                )
                torch.save(state, ckpt_best)
