import torch
from PIL import Image
from diffusers import DiffusionPipeline
import cv2
import numpy as np
from einops import rearrange
import os

def resize_image(image_path, output_size=(224, 224)):
    image = Image.open(image_path)
    target_aspect = output_size[0] / output_size[1]
    image_aspect = image.width / image.height

    if image_aspect > target_aspect:
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

def convert_to_tensor(data):
    # 如果 data 是 PIL.Image 对象，转换为张量
    if isinstance(data, Image.Image):
        return torch.tensor(np.array(data)).permute(2, 0, 1).float() / 255.0
    # 如果 data 是列表，递归地将其转换为张量
    if isinstance(data, list):
        return torch.stack([convert_to_tensor(item) for item in data])
    return data

def generate_video(input_img_path, repo_id, output_video_path, device="cuda"):
    # 加载预训练模型
    token = "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"
    pipeline = DiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token = token,
    )

    pipeline.to(device)

    input_image = resize_image(input_img_path)

    # 转换为Tensor并标准化
    input_tensor = torch.unsqueeze(torch.tensor(np.array(input_image)).permute(2, 0, 1).float() / 255.0, 0).to(device)

    with torch.no_grad():
        video = pipeline(input_tensor, num_inference_steps=10 , decode_chunk_size=1)

    # 获取 video 中的 frames 字段
    video_frames = video.frames

    # 将每个帧列表元素转换为张量
    if isinstance(video_frames, list):
        video_frames = [convert_to_tensor(frame) for frame in video_frames]
        video_frames = torch.stack(video_frames)

    if video_frames.dim() == 5 and video_frames.size(0) == 1:
        video_frames = video_frames.squeeze(0)

    # 处理生成的视频帧
    frames = (rearrange(video_frames, 't c h w -> t h w c') * 255).cpu().numpy().astype(np.uint8)

    # 保存视频
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        6,  # FPS
        (frames.shape[2], frames.shape[1])
    )

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_img_path = os.path.join(current_dir,  "pic.png")
    output_video_path = os.path.join(current_dir,  "output_video.mp4")
    repo_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

    generate_video(input_img_path, repo_id,  output_video_path)


