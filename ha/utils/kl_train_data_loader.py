import os
import re
from typing import Any
import numpy as np
import torch
import cv2
import torchvision
from utils.ucf_crime_utils import read_annotation_json
from PIL import Image

## First, encode videos into tensors and store said tensors with annotations.
## Only keep chunks (of 32) together, disregard other orderings.


def load_video(video_path) -> tuple[Any, int]:
    """Loads the video and returns a frame iterator and metadata."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video at {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    return cap, total_frames


class TorchVideoEncoder:
    def __init__(self):
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
                ]
            )
        
    def _init_objects(self, output_dir: str, annotation_path: str, video_dir: str, batch_size: int = 2048):
        os.makedirs(output_dir, exist_ok=True)
        annotations = read_annotation_json(annotation_path)
        video_paths_with_annotation = [(video_dir + "/" + re.split(r'\d', annotation.video_name)[0] + "/" + annotation.video_name if not "Normal" in annotation.video_name else f"{video_dir}/Training_Normal_Videos_Anomaly/{annotation.video_name}", annotation) for annotation in annotations]
        counter = 0
        batch_number = 0
        torch_frames = torch.empty((batch_size, 3, 224, 224))
        torch_annotations = torch.empty((batch_size))
        return video_paths_with_annotation, counter, batch_number, torch_frames, torch_annotations
        
    def encode(self, output_dir: str, annotation_path: str, video_dir: str, batch_size: int = 2048):
        video_paths_with_annotation, counter, batch_number, torch_frames, torch_annotations = self._init_objects(output_dir, annotation_path, video_dir, batch_size)
        for video, annotation in video_paths_with_annotation:
            try:
                cap, total_frames = load_video(video)
            except:
                continue
            for i in range(total_frames):
                if counter == batch_size:
                    torch.save((torch_frames, torch_annotations), f"{output_dir}/train_{batch_number}.pt")
                    torch_frames = torch.empty((batch_size, 3, 224, 224,))
                    torch_annotations = torch.empty((batch_size))
                    counter = 0
                    batch_number += 1
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    break
                torch_frames[counter] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) # type: ignore
                torch_annotations[counter] =int(annotation.is_frame_anomalous(i))
                counter += 1
    
    def encode_normal(self, output_dir: str, annotation_path: str, video_dir: str, batch_size: int = 2048):
        torch_annotations = torch.zeros((batch_size))
        video_paths_with_annotation, counter, batch_number, torch_frames, _ = self._init_objects(output_dir, annotation_path, video_dir, batch_size)
        for video, annotation in video_paths_with_annotation:
            try:
                cap, total_frames = load_video(video)
            except:
                continue
            for i in range(total_frames):
                if counter == batch_size:
                    torch.save((torch_frames, torch_annotations), f"{output_dir}/train_{batch_number}.pt")
                    torch_frames = torch.empty((batch_size, 3, 224, 224,))
                    counter = 0
                    batch_number += 1
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    break
                if not annotation.is_frame_anomalous(i):
                    torch_frames[counter] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) # type: ignore
                    counter += 1
    
    def encode_anomaly(self, output_dir: str, annotation_path: str, video_dir: str, batch_size: int = 2048):
        torch_annotations = torch.ones((batch_size))
        video_paths_with_annotation, counter, batch_number, torch_frames, _ = self._init_objects(output_dir, annotation_path, video_dir, batch_size)
        for video, annotation in video_paths_with_annotation:
            try:
                cap, total_frames = load_video(video)
            except:
                continue
            for i in range(total_frames):
                if counter == batch_size:
                    torch.save((torch_frames, torch_annotations), f"{output_dir}/train_{batch_number}.pt")
                    torch_frames = torch.empty((batch_size, 3, 224, 224,))
                    counter = 0
                    batch_number += 1
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    break
                if annotation.is_frame_anomalous(i):
                    torch_frames[counter] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) # type: ignore
                    counter += 1
    
    def encode_train_avi_dir(self, output_dir: str, video_dir: str, batch_size: int = 2048):
        torch_annotations = torch.zeros((batch_size))
        torch_frames = torch.empty((batch_size, 3, 224, 224))
        counter = 0
        batch_number = 0
        for filename in os.listdir(video_dir):
            if filename.endswith(".avi"):
                filepath = os.path.join(video_dir, filename)
                try:
                    cap, total_frames = load_video(filepath)
                except:
                    continue
                for i in range(total_frames):
                    if counter == batch_size:
                        torch.save((torch_frames, torch_annotations), f"{output_dir}/train_{batch_number}.pt")
                        torch_frames = torch.empty((batch_size, 3, 224, 224,))
                        counter = 0
                        batch_number += 1
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        break
                    torch_frames[counter] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) # type: ignore
                    counter += 1

    def encode_test_avi_dir(self, output_dir: str, frames_dir: str, masks_dir: str, batch_size: int = 2048):
        torch_annotations = torch.empty((batch_size))
        torch_frames = torch.empty((batch_size, 3, 224, 224))
        counter = 0
        batch_number = 0
        for subdir_name in os.listdir(frames_dir):
            mask_path = os.path.join(masks_dir, subdir_name + ".npy")
            mask = torch.tensor(np.load(mask_path))
            subdir_path = os.path.join(frames_dir, subdir_name)
            if os.path.isdir(subdir_path):
                # Iterate through images in each subdirectory
                for i, filename in enumerate(os.listdir(subdir_path)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')): # Add more image extensions if needed
                        if counter == batch_size:
                            torch.save((torch_frames, torch_annotations), f"{output_dir}/train_{batch_number}.pt")
                            torch_frames = torch.empty((batch_size, 3, 224, 224))
                            counter = 0
                            batch_number += 1

                        frame_path = os.path.join(subdir_path, filename)
                         # Construct mask path

                        try:
                            # Load and process frame
                            frame = cv2.imread(frame_path)
                            if frame is None:
                                print(f"Error: Could not read image file {frame_path}")
                                continue
                            
                            torch_frames[counter] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) # type: ignore

                            # Load and process mask
                            torch_annotations[counter] = mask[i]
                            counter += 1

                        except FileNotFoundError:
                            print(f"Warning: Mask file not found for {frame_path}")
                        except Exception as e:
                            print(f"Error processing {frame_path}: {e}")







## Secondly, load tensors into dataloader. Write that dataloader.

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, batch_size: int = 2048, clip_length: int = 8, step_size: int = 8):
        self.dir = dir
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.files = os.listdir(dir)
        self.max_file_number = max(
            int(re.search(r'(train|test)_(\d+)\.pt$', file).group(2)) # type: ignore
            for file in self.files if re.search(r'(train|test)_(\d+)\.pt$', file)
        )
        self.current_file = None
        self.step_size = step_size

    def __len__(self):
        return (self.max_file_number * self.batch_size//self.clip_length)//self.step_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.step_size * self.clip_length
        file_number = start_idx // self.batch_size
        start_offset = start_idx % self.batch_size

        if file_number != self.current_file or self.current_file is None:
            self.current_file = file_number
            self.current_features, self.current_labels = torch.load(f'{self.dir}/{self.files[self.current_file]}')
        
        
        return_features = self.current_features[start_offset:start_offset+self.clip_length*self.step_size:self.step_size]
        return_labels = self.current_labels[start_offset:start_offset+self.clip_length*self.step_size:self.step_size]

        return return_features, return_labels
    