#!/usr/bin/env python3
# Adapted from Allegro's vae_inference.py script

from einops import rearrange
import torch
import os
import argparse
import time
from src.allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

from decord import VideoReader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def vae_inference(args):
    start = time.time()

    # vae have better performance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()
    vae = torch.compile(vae, mode="max-autotune")

    vr = VideoReader(args.input_video)

    # frames = vr.get_batch(range(len(vr))).asnumpy()
    # frames = torch.from_numpy(frames).float() / 255.0
    # frames = frames * 2.0 - 1.0
    # frames = rearrange(frames, 'f h w c -> 1 c f h w')

    frames = vr.get_batch(range(int(args.frames))).asnumpy()

    frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)

    frames = frames.cuda(non_blocking=True).to(torch.float32)

    frames.div_(255.0).mul_(2.0).sub_(1.0)

    frames = frames.contiguous(
        memory_format=torch.channels_last_3d
    )

    frames = frames.cuda().to(torch.float32)

    with torch.inference_mode():
        posterior = vae.encode(
            frames,
            local_batch_size=args.local_batch_size
        )

        latents = posterior.latent_dist.mean * vae.scale_factor

    print("Latent shape:", latents.shape)

    video_name = os.path.splitext(os.path.basename(args.input_video))[0]
    latent_path = os.path.join(args.save_path, f"{video_name}_latents.pt")

    torch.save(latents.cpu(), latent_path)
    print(f"Saved latents to {latent_path}")

    end = time.time()
    print(f'Elapsed: {end - start:.2f} seconds')

if __name__ == "__main__":

    # python scripts\vae_encode_save.py --input_video processed_data/room/traj_00001/render_traj_color.mp4

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default='models/allegro/vae')
    parser.add_argument("--frames", type=str, default=88)
    parser.add_argument("--input_video", type=str, default="data/room/traj_00001/render_traj_color.mp4")
    parser.add_argument("--save_path", type=str, default="latents")
    parser.add_argument("--local_batch_size", type=int, default=1)

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    vae_inference(args)