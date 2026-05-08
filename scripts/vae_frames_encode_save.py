#!/usr/bin/env python3

from einops import rearrange
import torch
import os
import argparse
import time
import numpy as np

from src.allegro.models.vae.vae_allegro import (
    AllegroAutoencoderKL3D
)

from decord import VideoReader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def vae_inference(args):

    start = time.time()

    vae = AllegroAutoencoderKL3D.from_pretrained(
        args.vae,
        torch_dtype=torch.float32
    ).cuda()

    vae = torch.compile(
        vae,
        mode="max-autotune"
    )

    vr = VideoReader(args.input_video)
    total_frames = len(vr)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        args.num_frames,
        dtype=int
    )

    print("Selected frames:")
    print(frame_indices)

    video_name = os.path.splitext(
        os.path.basename(args.input_video)
    )[0]

    for frame_idx in frame_indices:

        print(f"\nEncoding frame {frame_idx}")

        frame = vr[frame_idx].asnumpy()

        frame = torch.from_numpy(frame)

        # HWC -> CHW
        frame = frame.permute(2, 0, 1)

        # duplicate single frame across time dimension
        T = 8

        frame = (
            frame
            .unsqueeze(1)          # (C,1,H,W)
            .repeat(1, T, 1, 1)    # (C,T,H,W)
        )

        # add batch dimension
        frame = frame.unsqueeze(0)

        # final shape:
        # (1,C,T,H,W)

        frame = frame.cuda(
            non_blocking=True
        ).to(torch.float32)

        with torch.inference_mode():

            posterior = vae.encode(
                frame,
                local_batch_size=args.local_batch_size
            )

            latents = (
                posterior.latent_dist.mean
                * vae.scale_factor
            )

        print("Latent shape:", latents.shape)

        latent_path = os.path.join(
            args.save_path,
            f"{video_name}_{frame_idx}.pt"
        )

        torch.save(
            latents.cpu(),
            latent_path
        )

        print(f"Saved -> {latent_path}")

    end = time.time()

    print(f"\nElapsed: {end - start:.2f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vae",
        type=str,
        default="models/allegro/vae"
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=8
    )

    parser.add_argument(
        "--input_video",
        type=str,
        default="data/room/traj_00001/render_traj_color.mp4"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="latents_f"
    )

    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=1
    )

    args = parser.parse_args()

    os.makedirs(
        args.save_path,
        exist_ok=True
    )

    vae_inference(args)