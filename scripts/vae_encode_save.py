# Adapted from Allegro's vae_inference.py script
# 

from einops import rearrange
import torch
import os
import argparse
from src.allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

from decord import VideoReader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def vae_inference(args):

    # vae have better performance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()

    vr = VideoReader(args.input_video)

    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames * 2.0 - 1.0
    frames = rearrange(frames, 'f h w c -> 1 c f h w')
    frames = frames[:,:,:88]

    frames = frames.cuda().to(torch.float32)

    with torch.no_grad():
        posterior = vae.encode(
            frames,
            local_batch_size=args.local_batch_size
        )

        latents = posterior.latent_dist.sample() * vae.scale_factor

    print("Latent shape:", latents.shape)

    video_name = os.path.splitext(os.path.basename(args.input_video))[0]
    latent_path = os.path.join(args.save_path, f"{video_name}_latents.pt")

    torch.save(latents.cpu(), latent_path)
    print(f"Saved latents to {latent_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default='./models/allegro/vae')
    parser.add_argument("--input_video", type=str, default="./resources/render_traj_color.mp4")
    parser.add_argument("--save_path", type=str, default="./allegro_latents")
    parser.add_argument("--local_batch_size", type=int, default=2)

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    vae_inference(args)