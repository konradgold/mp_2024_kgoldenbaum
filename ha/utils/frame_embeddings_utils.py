import torch
import numpy as np
from PIL import Image
import os
import time
import cv2
from tqdm import tqdm

LABEL_TO_INDEX = {'Arson' : 1, 
                  'Burglary' : 2, 
                  'Shooting': 3, 
                  'Robbery': 4, 
                  'Abuse': 5, 
                  'Explosion' : 6, 
                  'Training_Normal_Videos_Anomaly': 7, 
                  'Vandalism': 8, 
                  'Banner': 9, 
                  'RoadAccidents':10, 
                  'Shoplifting': 11, 
                  'MolotovBomb': 12, 
                  'Assault': 13, 
                  'Stealing': 14, 
                  'Arrest': 15, 
                  'Fighting': 16,
                  'Normal': 0,
                  'Text': -1
                  }

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

def _get_frame_embeddings(frames, embedder, number_per_batch, interval: int = 1):
    """
    Returns embeddings for each frame in the video using the given embedder model.
    """

    results = []

    current_frame = 0
    images = []
    for frame in frames:
        if current_frame % interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(images) < number_per_batch:
                images.append(pil_image)
            else:
                embeddings = embedder.extract(images)
                results += embeddings
                images = []

            current_frame += 1
    return results

def _get_frames(video_path: str, interval: int = 1):
    """
    Returns frames for each frame in the video.
    """
    timeout: int | None = None  # seconds
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    progress_bar = tqdm(total=total_frames, desc="Analyzing Frames", unit="frame")

    start_time = time.time()
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % interval == 0:
            yield frame

            current_frame += 1
            progress_bar.update(1)

        if timeout and time.time() - start_time > timeout:
            break

    cap.release()
    progress_bar.close()


def embed_video(video, embedder, number_per_batch=15):
    """
    Gets all segments of a video as file paths and returns the embeddings for each frame in the whole video,
    as well as the labels and info (segment file name + frame number) for each frame.
    """
    result = []

    result = _get_frame_embeddings(video, embedder, interval=1, number_per_batch=number_per_batch)

    # print(f"Length of results: {len(results)}")

    if len(result) == 0:
        return None, None, None

    result = np.concatenate(result, axis=0)
    result = torch.tensor(result)
    if len(result.size()) == 3:
        result = result.squeeze(1)
    return result


def save_data(embeddings, labels, annotations, model_name, video_name):
    """
    Saves the embeddings, labels and annotations for a video in a directory ('embeddings').
    """
    directory = "embeddings"
    if not os.path.exists(directory):
        os.makedirs(directory)

    subdirectory = f"{directory}/{video_name}"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    np.save(f"{subdirectory}/{model_name}_embeddings.npy", embeddings)
    np.save(f"{subdirectory}/{model_name}_labels.npy", labels)
    np.save(f"{subdirectory}/{model_name}_annotations.npy", annotations)


def load_data(model_name, video_name, frame_steps=1) -> tuple[np.ndarray | None, np.ndarray| None, list[tuple] | None]:
    """
    Loads the embeddings, labels and annotations for a video from the 'embeddings' directory.
    """
    directory = "//mnt/ssd2/embeddings"
    subdirectory = f"{directory}/{video_name}"

    if not os.path.exists(subdirectory):
        print(f"Embeddings for {video_name} not found")
        return None, None, None

    embeddings = np.load(f"{subdirectory}/{model_name}_embeddings.npy")[::frame_steps] # (n, embed_dim)
    labels = np.load(f"{subdirectory}/{model_name}_labels.npy")[::frame_steps] # (n,)
    annotations = np.load(f"{subdirectory}/{model_name}_annotations.npy")[::frame_steps] # list(n,) 

    # Convert annotations to list of tuples
    annotations = [tuple(ann) for ann in annotations]

    return embeddings, labels, annotations


def get_all_video_names(directory: str, exclude_normal=False, filter_list = []):
    """
    Gets all video names in the given directory.
    """
    video_names = []
    for _, dirs, _ in os.walk(directory):
        for dir in dirs:
            if dir.endswith(".mp4"):
                if exclude_normal and "Normal_Video" in dir:
                    continue
                video_names.append(dir)
    
    if len(filter_list) > 0:
        video_names = [video for video in video_names if video in filter_list]

    return video_names


def get_video_names_of_class(directory: str, class_name: str):
    """
    Gets all video names in the given directory that belong to the given class.
    """
    all_videos = get_all_video_names(directory)
    return [video for video in all_videos if video.startswith(class_name)]


def load_multiple_embeddings(model_name: str, video_names: list[str], frame_steps=1):
    """
    Loads the embeddings, labels and annotations for all videos in the given list of video names from a given
    embedding model.
    """
    # Load embeddings, labels and annotations for all videos and concatenate them into a single numpy array
    all_embeddings = []
    all_labels = []
    all_annotations = []

    for video_name in tqdm(video_names):
        embeddings, labels, annotations = load_data(model_name, video_name, frame_steps=frame_steps)
        if embeddings is not None:
            all_embeddings.append(embeddings)
            all_labels.append(labels)
            all_annotations.extend(annotations)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings, all_labels, all_annotations


def load_embeddings_in_chunks(model_name: str, video_names: list[str], frame_steps=1, shuffle=True):
    """
    Loads the embeddings, labels and annotations for all videos and yields them in chunks.
    """
    if shuffle:
        video_names = np.random.permutation(video_names)

    for video_name in video_names:
        embeddings, labels, annotations = load_data(model_name, video_name, frame_steps=frame_steps)
        if embeddings is None:
            continue
        yield embeddings, labels, annotations

