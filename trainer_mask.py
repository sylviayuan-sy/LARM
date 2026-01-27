import os
import sys
import torch
import argparse
import yaml
import wandb
import torch.distributed as tdist
import torch.utils.data as tdata
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr_compute

from model.larm_mask import LARM
from data.data_mask import Dataset
from model.loss_mask import Loss


def init_process(args):
    tdist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Running process {rank}:{local_rank}")
    return rank, local_rank


def init_model(model_config, training_config, dataset_config, local_rank):
    part_finetune = dataset_config["part_finetune"]
    patch_size = model_config["patch_size"]
    model = LARM(
        model_config["hidden"],
        model_config["num_layers"],
        patch_size * patch_size * 9 + (1 if part_finetune else 0),
        patch_size * patch_size * 6 + (1 if part_finetune else 0),
        patch_size,
        model_config["linear_dim"],
        dataset_config["resolution"],
        dataset_config["num_target_views"],
        dataset_config["num_input_views"],
        training_config["batch_size"]
    ).cuda()
    model = DDP(model, device_ids=[local_rank])

    if training_config["resume_ckpt"]:
        pretrained_model = torch.load(training_config["resume_ckpt"], weights_only=True)
        curr_state_dict = model.state_dict()
        for key in curr_state_dict:
            try:
                if key != "module.decoder.input_linear.weight" and key != "module.decoder.target_linear.weight" and key != "module.decoder.output_linear.weight":
                    curr_state_dict[key] = pretrained_model[key].clone()
                elif key == "module.decoder.output_linear.weight":
                    curr_state_dict[key][:pretrained_model[key].shape[0], :] = pretrained_model[key].clone()
                else:
                    curr_state_dict[key][:, :pretrained_model[key].shape[1]] = pretrained_model[key].clone()
            except:
                pass
        model.load_state_dict(curr_state_dict)
        del pretrained_model

    return model


def init_optimizer_scheduler(model, training_config):
    layernorm_params = [p for name, p in model.named_parameters() if 'norm_' in name]
    other_params = [p for name, p in model.named_parameters() if 'norm_' not in name and "module.decoder.input_linear.weight" not in name and "module.decoder.target_linear.weight" not in name]
    token_params = [p for name, p in model.named_parameters() if "module.decoder.input_linear.weight" in name or "module.decoder.target_linear.weight" in name]
    optimizer = torch.optim.AdamW([
                {'params': other_params, 'name':"other"},
                {'params': token_params, 'name':"token"},
                {'params': layernorm_params, 'weight_decay': 0.0, 'name':"norm"}
                ], lr=training_config["lr"], betas=(training_config["beta1"], training_config["beta2"]), weight_decay=training_config["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_config["warmup_steps"],
            num_training_steps=training_config["num_iters"],
        )
    return optimizer, scheduler


def init_dataloaders(dataset_config, training_config):
    train_set = Dataset(
        dataset_config["train_data"],
        dataset_config["num_input_views"] + dataset_config["num_target_views"],
        dataset_config["resolution"],
        part_finetune=dataset_config["part_finetune"]
    )
    eval_set = Dataset(
        dataset_config["eval_data"],
        dataset_config["num_input_views"] + dataset_config["num_target_views"],
        dataset_config["resolution"],
        part_finetune=dataset_config["part_finetune"]
    )

    train_loader = tdata.DataLoader(
        train_set,
        batch_size=training_config["batch_size"],
        shuffle=False,
        sampler=tdata.DistributedSampler(train_set),
        num_workers=training_config["num_data_workers"],
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    eval_loader = tdata.DataLoader(
        eval_set,
        batch_size=training_config["eval_batch_size"],
        shuffle=False,
        sampler=tdata.DistributedSampler(eval_set, shuffle=False),
        num_workers=training_config["num_data_workers"],
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, eval_loader


def log_to_wandb(loss_dict: dict, prefix: str = "train", extra_metrics: dict = None):
    log_data = {}
    for key, val in loss_dict.items():
        log_data[f"{prefix}/{key}"] = val.item() if torch.is_tensor(val) else val
    if extra_metrics:
        for key, val in extra_metrics.items():
            log_data[f"{prefix}/{key}"] = val.item() if torch.is_tensor(val) else val
    wandb.log(log_data)


@torch.no_grad()
def eval_loop(model, eval_loader, criterion, dataset_config, training_config, rank, local_rank):
    model.eval()
    
    psnr_sum = torch.tensor(0.0, device=f"cuda:{local_rank}")
    psnr_count = torch.tensor(0, device=f"cuda:{local_rank}")
    total_loss = torch.tensor(0.0, device=f"cuda:{local_rank}")
    num_batches = torch.tensor(0, device=f"cuda:{local_rank}")

    for data in eval_loader:
        intrinsic = data["fxfycxcy"].cuda(non_blocking=True)
        extrinsic_in = data["c2w"][:, :dataset_config["num_input_views"]].cuda(non_blocking=True)
        extrinsic_out = data["c2w"][:, dataset_config["num_input_views"]:].cuda(non_blocking=True)
        image = data["image"].cuda(non_blocking=True)
        qpos_in = data["qpos"][:, :dataset_config["num_input_views"]].cuda(non_blocking=True) if dataset_config["part_finetune"] else None
        qpos_out = data["qpos"][:, dataset_config["num_input_views"]:].cuda(non_blocking=True) if dataset_config["part_finetune"] else None
        mask = data["mask"].cuda(non_blocking=True) if dataset_config["part_finetune"] else None
        depth = data["depth"].cuda(non_blocking=True) if dataset_config["part_finetune"] else None

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(intrinsic, extrinsic_in, extrinsic_out, image, qpos_in=qpos_in, qpos_out=qpos_out)
            out_perm = output.permute(0, 1, 4, 2, 3)

            if dataset_config["part_finetune"]:
                loss_dict, _ = criterion(out_perm, image[:, dataset_config["num_input_views"]:], mask, depth, intrinsic, torch.cat([extrinsic_in, extrinsic_out], dim=1))
            else:
                loss_dict, _ = criterion(out_perm, image[:, dataset_config["num_input_views"]:], None, None, None, None)

        batch_psnr = psnr_compute(out_perm[:, dataset_config["num_input_views"]:, :3].float(), image[:, dataset_config["num_input_views"]:].float())
        psnr_sum += batch_psnr.sum()
        psnr_count += batch_psnr.numel()
        total_loss += loss_dict["loss"].float()
        num_batches += 1

    tdist.all_reduce(psnr_sum, op=tdist.ReduceOp.SUM)
    tdist.all_reduce(psnr_count, op=tdist.ReduceOp.SUM)
    tdist.all_reduce(total_loss, op=tdist.ReduceOp.SUM)
    tdist.all_reduce(num_batches, op=tdist.ReduceOp.SUM)

    if rank == 0:
        psnr_mean = (psnr_sum / psnr_count).item()
        avg_loss = (total_loss / num_batches).item()
        log_to_wandb({"loss": avg_loss}, prefix="val", extra_metrics={"psnr": psnr_mean})

    tdist.barrier()
    model.train()


def run(config, rank, local_rank):
    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]
    loss_config = config["loss"]
    wandb_config = config["wandb"]

    if wandb_config.get("key"):
        wandb.login(key=wandb_config["key"])
    if rank == 0:
        wandb.init(
            project=wandb_config["project"],
            id=wandb_config.get("run_id"),
            resume="must" if wandb_config.get("run_id") else None
        )

    model = init_model(model_config, training_config, dataset_config, local_rank)
    optimizer, scheduler = init_optimizer_scheduler(model, training_config)
    criterion = Loss(loss_config["weight"], loss_config["depth_weight"], f"cuda:{local_rank}")
    train_loader, eval_loader = init_dataloaders(dataset_config, training_config)

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    num_iter = training_config.get("resume_iter") or 0
    train_iter = iter(train_loader)

    while num_iter < training_config["num_iters"]:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        intrinsic = data["fxfycxcy"].cuda(non_blocking=True)
        extrinsic_in = data["c2w"][:, :dataset_config["num_input_views"]].cuda(non_blocking=True)
        extrinsic_out = data["c2w"][:, dataset_config["num_input_views"]:].cuda(non_blocking=True)
        image = data["image"].cuda(non_blocking=True)
        qpos_in = data["qpos"][:, :dataset_config["num_input_views"]].cuda(non_blocking=True) if dataset_config["part_finetune"] else None
        qpos_out = data["qpos"][:, dataset_config["num_input_views"]:].cuda(non_blocking=True) if dataset_config["part_finetune"] else None
        mask = data["mask"].cuda(non_blocking=True) if dataset_config["part_finetune"] else None
        depth = data["depth"].cuda(non_blocking=True) if dataset_config["part_finetune"] else None

        if mask is not None:
            mask[mask >= 0.5] = 1.0
            mask[mask < 0.5] = 0.0

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(intrinsic, extrinsic_in, extrinsic_out, image, qpos_in=qpos_in, qpos_out=qpos_out)
            out_perm = output.permute(0, 1, 4, 2, 3)

            if dataset_config["part_finetune"]:
                loss_dict, _ = criterion(out_perm, image[:, dataset_config["num_input_views"]:], mask, depth, intrinsic, torch.cat([extrinsic_in, extrinsic_out], dim=1))
            else:
                loss_dict, _ = criterion(out_perm, image[:, dataset_config["num_input_views"]:], None, None, None, None)

        scaler.scale(loss_dict["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if rank == 0 and num_iter % training_config["log_interval"] == 0:
            extra = {
                "learning_rate": scheduler.get_last_lr()[0],
                "psnr": psnr_compute(out_perm[:, dataset_config["num_input_views"]:, :3], image[:, dataset_config["num_input_views"]:])
            }
            log_to_wandb(loss_dict, prefix="train", extra_metrics=extra)

        if training_config.get("eval_interval") and num_iter % training_config["eval_interval"] == 0 and num_iter > 0:
            eval_loop(model, eval_loader, criterion, dataset_config, training_config, rank, local_rank)

        num_iter += 1

    eval_loop(model, eval_loader, criterion, dataset_config, training_config, rank, local_rank)


def main(config, args):
    torch.cuda.empty_cache()
    rank, local_rank = init_process(args)
    tdist.barrier()
    run(config, rank, local_rank)
    tdist.barrier()
    tdist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    os.makedirs(config["training"]["vis_dir"], exist_ok=True)

    main(config, args)
