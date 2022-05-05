# CS153 Shark and Human Tracking Data Processing Pipeline

Please see the warning at the end of the example function calls before using the data procesessing pipeline. 
## Run the entire workflow

In the terminal, you can type the following: 

    python3 main.py 
        -v [path for VIA jsons directory] 
        -d [path for video directory] 
        -c [path for the directory to store the COCO jsons] 
        -m [file path to store the merged COCO json] 
        -f [path for the directory to store frames] 
        -a [file path to store the filename mapping]

A specific example is:

    python3 main.py -v "./via_annotations/" -d "./videos/" -c "./coco_annotations/" -m "/merged_coco_annotation/merged_coco.json" -f "./frames/" -a "./video_file_id_map.json"


## Run the main sections of the workflow individually

### Convert ALL VIA annotations to individual COCO annotations

From the `via2CocoConverter.py`, you can run
```python
convertAllViaToCoco(via_json_dir, video_dir, coco_json_dir)
```

If any VIA annotation encounters any error during the conversion, the VIA annotation's filename, the associated file id, and the error will be saved as a log file called `'via2coco_error_log.txt'` in the logs directory specified by the configuration file.

### Merge ALL COCO annotations to ONE COCO annotation

From the `via2CocoConverter.py`, you can run
```python
mergeAllCoco(coco_json_dir, merged_save_path)
```

If any COCO annotation encounters any error during the merging, the COCO annotation's filename and the error will be saved as a log file called `'cocomerge_error_log.txt'` in the logs directory specified by the configuration file.

### Convert ALL videos to frames

From the `video2FrameConverter.py`, you can run
```python
convertAllVideosToFrames(via_json_dir, video_dir, video_frame_dir)
```

If any video encounters any error during the conversion, the video's filename, the path to save the video's frame, and the error will be saved as a log file called `'video2frame_error_log.txt'` in the logs directory specified by the configuration file.


### Save video filenames and their associated file id

From the `video2FrameConverter.py`, you can run
```python
generatetVidToFileIdMap(via_json_dir, map_json_save_path)
```

It saves aa json with two main fields:
- "filenames": a list that contain all the video filenames (without the extension)
- "id_map": a dictionary that uses the video filenames as the key and match the video filenames to a list of file ids that are associated with the video.

## Run functions on individual function
### Convert ONE VIA annotations to ONE COCO annotations

From the `via2CocoConverter.py`, you can run
```python
convertToCocoFormat(via_json_path, video_dir, coco_json_dir, file_id)
```

### Merge a pair of TWO COCO annotations to ONE COCO annotation

From the `./merge_coco/merge.py`, you can run
```python
combine(tt1, tt2, output_file)
```

### Convert a video to frames

From the `video2FrameConverter.py`, you can run
```python
convertVideoToFrame(video_path, video_frame_path)
```

## Warning

WARNING: we do have to make the following assumptions in order for this to run smoothly:

- A given VIA annotation json has the same filename as the corresponding video filename. 
If there exists another VIA annotation json for the same video, it will get appended ``\_2" at the end of the filename.
There are at most two VIA annotation json for the same video.

- All video files are in \.mp4 format.

- The user has already initialized directories to store the conversion results. 
For example, the user has already created empty folders for saving the converted COCO annotation jsons, the merged, overall COCO annotation json, and the frames. 