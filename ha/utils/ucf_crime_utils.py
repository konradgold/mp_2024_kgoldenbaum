import os
import json
import cv2
import decord
import numpy as np


class VideoAnnotationInfo:

    def __init__(self, video_name, label, starting_frames, ending_frames, skip = 1):
        self.video_name = video_name
        self.label = label
        self.starting_frames = starting_frames
        self.ending_frames = ending_frames
        self.skip = skip if skip is not None else 1

    def __str__(self) -> str:
        return f"Video: {self.video_name}, Label: {self.label}, Start: {self.starting_frames}, End: {self.ending_frames}"
 
    def is_frame_anomalous(self, frame_number) -> bool:
        frame_number = frame_number * self.skip
        for start, end in zip(self.starting_frames, self.ending_frames):
            if start <= frame_number <= end:
                return True
        return False

    def set_skip(self, skip):
        self.skip = skip
        return self


def read_annotation_text(annotation_path) -> list[VideoAnnotationInfo]:
    videos = []

    with open(annotation_path, "r") as file:
        for line in file.readlines():
            line = line.strip().split("  ")
            video_name = line[0]
            label = line[1]
            start1 = int(line[2])
            end1 = int(line[3])
            start2 = int(line[4])
            end2 = int(line[5]) 

            starting_frames = [start1, start2]
            ending_frames = [end1, end2]

            # Dont save start2 and end2 if they are -1
            if start2 == -1 and end2 == -1:
                starting_frames = [start1]
                ending_frames = [end1]
            
            videos.append(
                VideoAnnotationInfo(
                    video_name=video_name,
                    label=label,
                    starting_frames=starting_frames,
                    ending_frames=ending_frames)
            )
           
    return videos


def read_annotation_json(annotation_path) -> list[VideoAnnotationInfo]:
    videos = []

    with open(annotation_path, "r") as file:
        data = json.load(file)
        for key, video_data in data.items():
            label = key.split("/")[0]
            video_name = key.split("/")[1]

            starting_frames = [x["start"] for x in video_data]
            ending_frames = [x["end"] for x in video_data]

            videos.append(
                VideoAnnotationInfo(
                    video_name=video_name,
                    label=label,
                    starting_frames=starting_frames,
                    ending_frames=ending_frames)
            )
            
    return videos


def split_video(
        input_path, 
        output_dir, 
        segment_duration=10, 
        start_from_frame=0,
        end_at_frame=None) -> None:
    """
    Split an MP4 video into {segment_duration}-second segments.
    If a segment is not exactly {segment_duration} seconds, it will be shorter.
    
    Args:
        input_path (str): Path to the input MP4 video file
        output_dir (str): Directory to save split video segments
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame count for x-second segments
    segment_frames = int(fps * segment_duration)
    
    # Video writer setup variables
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Process video
    current_segment = 0
    frame_count = 0
    output_video = None
    
    while True:
        ret, frame = video.read()
        
        # Break if no more frames
        if not ret:
            break

        # Skip frames until start_from_frame
        if frame_count < start_from_frame:
            frame_count += 1
            continue
        
        # Start a new segment video writer if needed
        if frame_count % segment_frames == 0:
            # Close previous video writer if it exists
            if output_video:
                output_video.release()
            
            # Create new output filename
            output_filename = os.path.join(
                output_dir, 
                f'segment_{current_segment}.mp4'
            )
            
            # Create new video writer
            output_video = cv2.VideoWriter(
                output_filename, 
                fourcc, 
                fps, 
                (frame.shape[1], frame.shape[0])
            )
            
            current_segment += 1
        
        # Write frame to current segment
        output_video.write(frame) # type: ignore
        frame_count += 1

        # Break if end_at_frame is reached
        if end_at_frame and frame_count >= end_at_frame:
            break
    
    # Release final video writer and video capture
    if output_video:
        output_video.release()
    video.release()
    
    print(f"Video split into {current_segment} segments of {segment_duration} seconds each.")


def extract_video_segments(input_path, output_dir, start_frames, end_frames) -> None:
    """
    Extract video segments from specified frame ranges and save them as separate MP4 files.
    
    Args:
        input_path (str): Path to the input video file
        output_dir (str): Directory to save output video segments
        start_frames (list): List of start frames for each segment
        end_frames (list): List of end frames for each segment
    
    Raises:
        ValueError: If start_frames and end_frames have different lengths
        ValueError: If any frame range is invalid
    """
    # Remove -1
    start_frames = [x for x in start_frames if x != -1]
    end_frames = [x for x in end_frames if x != -1]

    # Remove if a segment is shorter than 2 frames
    differences = [end_frames[i] - start_frames[i] for i in range(len(start_frames))]
    if any([x < 2 for x in differences]):
        tmp_start_frames = []
        tmp_end_frames = []
        
        for i in range(len(start_frames)):
            if differences[i] < 2:
                continue
            tmp_start_frames.append(start_frames[i])
            tmp_end_frames.append(end_frames[i])
        
        start_frames = tmp_start_frames
        end_frames = tmp_end_frames

    # Validate input
    if len(start_frames) != len(end_frames):
        raise ValueError("Start and end frame lists must have the same length")
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # If no start frames are provided, extract the entire video
    if len(start_frames) == 0:
        start_frames = [0]
        end_frames = [total_frames - 1]

    # Extract segments
    for i, (start, end) in enumerate(zip(start_frames, end_frames), 1):
        # Validate frame ranges
        if start < 0 or end >= total_frames or start > end:
            raise ValueError(f"Invalid frame range: start={start}, end={end}, total_frames={total_frames}")
        
        # Prepare output video writer
        output_path = os.path.join(output_dir, f'segment_{i}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Reset video capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        # Create video writer
        out = None
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Stop if no more frames or reached end frame
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end + 1:
                break
            
            # Initialize video writer on first valid frame
            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frame
            out.write(frame)
            frame_count += 1
        
        # Close writer for this segment
        if out:
            out.release()
    
    # Release input video capture
    cap.release()


def get_fps_and_frame_count(video_path) -> tuple[int, int]:
    """
    Get the frame rate and total frame count of a video file.
    
    Args:
        video_path (str): Path to the input video file
    
    Returns:
        tuple: A tuple containing the frame rate and total frame count
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def get_video_inputs_opencv(video_path, frames_per_clip=8, every_nth_frame=30):
    """
    Sample frames_per_clip * every_nth_frame long segments from the video.
    If every_nth_frame equals fps of the video, then this function samples 1 frame per second and
    one segment is equal to 8 seconds of video.
    """
    video = cv2.VideoCapture(video_path)
    segments = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_length = frames_per_clip * every_nth_frame

    for segment_start in range(0, frame_count - 1, segment_length):
        segment = []
        for frame_no in range(segment_start, min(frame_count - 1, segment_start + segment_length), every_nth_frame):
            video.set(1, frame_no)
            _, frame = video.read()
            segment.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        segments.append(segment)
    video.release()
    return segments


import cv2
import decord
import numpy as np

def get_video_inputs_decord(video_path, frames_per_clip=8, every_nth_frame=30, verbose=False):
    """
    Sample frames_per_clip * every_nth_frame long segments from the video.
    If every_nth_frame equals fps of the video, then this function samples 1 frame per second and
    one segment is equal to 8 seconds of video.
    """

    def _print_verbose(msg):
        if verbose:
            print(msg)

    video = cv2.VideoCapture(video_path)
    segments = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    segment_length = frames_per_clip * every_nth_frame
    decord_vr = decord.VideoReader(video_path, ctx=decord.cpu(0))

    _print_verbose(f"Frame count: {frame_count}")
    _print_verbose(f"Segment length: {segment_length}")
    _print_verbose("Sampled frames: ")

    assert frame_count > 0, "Frame count is 0"

    # Video is shorter than segment_length
    if frame_count < segment_length: 
        frame_ids = np.linspace(0, frame_count-1, frames_per_clip, dtype=int)
        segments.append(decord_vr.get_batch(frame_ids))
        _print_verbose(frame_ids)
        return segments
    
    for segment_start in range(0, frame_count, segment_length):
        frame_ids =  list(range(segment_start, min(frame_count - 1, segment_start + segment_length), every_nth_frame))
        if len(frame_ids) == frames_per_clip:
            segments.append(decord_vr.get_batch(frame_ids))
            _print_verbose(frame_ids)
    
    # Last segment might be shorter
    if frame_count - segment_start > frames_per_clip: # type: ignore
        frame_ids = np.linspace(segment_start, frame_count-1, frames_per_clip, dtype=int) # type: ignore
        segments.append(decord_vr.get_batch(frame_ids))
        _print_verbose(frame_ids)

    return segments