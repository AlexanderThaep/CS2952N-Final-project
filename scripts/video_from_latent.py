#!/usr/bin/env python3

import torch
import imageio
import os
import argparse

from src.allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def vae_decode(args):

    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()
    vae = torch.compile(vae, mode="max-autotune")

    latents = torch.load(args.latents, map_location="cuda").to(torch.float32)

    with torch.inference_mode():
        out_video = vae.decode(
            latents / vae.scale_factor,
            local_batch_size=args.local_batch_size
        ).sample

        out_video = (
            (out_video / 2.0 + 0.5)
            .clamp(0, 1)
            .mul(255)
            .to(dtype=torch.uint8)
            .cpu()
            .permute(0, 2, 3, 4, 1)
            .contiguous()
        )

    video_name = os.path.splitext(os.path.basename(args.input_video))[0]

    imageio.mimwrite(
        f"{args.save_path}/{video_name}.mp4",
        out_video[0],
        fps=15,
        quality=8
    )

if __name__ == "__main__":

    # python scripts\video_from_latent.py --latents latents/render_traj_color_latents.pt

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default='models/allegro/vae')
    parser.add_argument("--latents", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="output_videos")
    parser.add_argument("--local_batch_size", type=int, default=1)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    vae_decode(args)