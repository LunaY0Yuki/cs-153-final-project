import subprocess
import os
import traceback
import json

"""
Constant declaration (from config file)
"""
with open("./config.json", "r") as f:
  config_json = json.load(f)

LOGS_DIR = config_json["logs_dir"]

"""
====================================================================================================

    Main functions to run
      - convertAllVideosToFrames
          if you want to convert ALL the videos in directory to frames
      - convertToCocoFormat
          if you want to convert ONE video to frames
      - generatetVidToFileIdMap
          if you want to generate the json that will make the video filenames to their file ids 
====================================================================================================
"""

"""
Convert all the videos in {video_dir} to frames based on the via_json_dir filenames

Note that
    if there are two annotations for the video, we actually convert the video frame twice
        one set of the video frames will have _2 in the filename

Warning:
    All the directory path should end with "/"

Parameters:
    via_json_dir - string, directory containing all the via jsons
        WARNING:
        - we assume that the via annotation jsons have the same filename as the corresponding vidoe filename
        - if there exist another via annotation jsons for the same video, 
            the via annotation json would just have "_2" at the end
    video_dir - string, directory containing all the videos
        Warning: 
        - we assume that all video ends in .mp4
    video_frame_dir - string, directory where we would save the frames
"""
def convertAllVideosToFrames(via_json_dir, video_dir, video_frame_dir):
    frame_filename_list = []

    for _, _, filenames in os.walk(via_json_dir):
        # sort ascending
        filenames_sorted = sorted(filenames)

        for f in filenames_sorted:
            if os.path.splitext(f)[1] == ".json":
                
                if "_2" in f:
                    video_filename, _ = f.split("_2")
                else:
                    video_filename = os.path.splitext(f)[0]

                video_filename += ".mp4"
                frame_filename = os.path.splitext(f)[0] + "_%05d.jpg"
                
                # save both path to the actual video file and the path to save the frames
                frame_filename_list.append((os.path.join(video_dir, video_filename), os.path.join(video_frame_dir, frame_filename)))

    video_with_error = []

    for video_path, video_frame_path in frame_filename_list: 
        try: 
            convertVideoToFrame(video_path, video_frame_path)
            print()

        except:
            trace_error = traceback.format_exc()

            print(trace_error)

            print("xxxxxxxxxxx WARNING! xxxxxxxxxxx")
            print(f"xxxxxxx video path = {video_path} xxxxxxx")
            print(f"xxxxxxx video frame path = {video_frame_path} xxxxxxx")
            print()
            
            video_with_error.append((video_path, video_frame_path, trace_error))

            pass

    if video_with_error != []:
        with open(f"{os.path.join(LOGS_DIR, 'video2frame_error_log.txt')}", "w") as f:
            print("------------ List of Videos that Have Errors When Converting To Frames ------------")
            print("     format: (video path, video frame path)")

            f.write("------------ List of Videos that Have Errors When Converting To Frames ------------\n")
            f.write("     format: (video path, video frame path)\n")
            for video_path, video_frame_path, trace_error in video_with_error:
                print((video_path, video_frame_path))
                print(trace_error)
                print()

                f.write(f"{(video_path, video_frame_path)}\n")
                f.write(f"{trace_error}\n\n")
                      


"""
Convert ONE videos, specified in {video_path}, to frames

Parameters:
    video_path - string, path to the video file
    video_frame_path - string, path to save the frames
        Warning: frames are saved as .jpg
"""
def convertVideoToFrame(video_path, video_frame_path):
    print(f"Converting video = {video_path}")
    print(f" To frame filenames = {video_frame_path}")

    # Source on how to run shell scripts in python: https://janakiev.com/blog/python-shell-commands/
    process = subprocess.Popen(['ffmpeg',  '-i', video_path, 
                                            '-r', '10', 
                                            '-start_number', '0', 
                                            video_frame_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE,
                     universal_newlines=True)

    stdout, stderr = process.communicate()
    printStdOutput(stdout)
    printStdOutput(stderr)


"""
Generate a json that keeps track of
    -> a list with all the video filenames
    -> a directionary that map a particular video filename to the file ids that correspond to the video
        Note: a video could be associated with multiple file ids 
            (because multiple instances of annotation of that video could exist)

Parameters:
    via_json_dir - string, directory containing all the via jsons
        WARNING:
        - we assume that the via annotation jsons have the same filename as the corresponding vidoe filename
        - if there exist another via annotation jsons for the same video, 
            the via annotation json would just have "_2" at the end
    map_json_save_path - string, path (must include the filename) where we save the mapping
"""
def generatetVidToFileIdMap(via_json_dir, map_json_save_path):
    video_file_id_map = {"filenames": [], "id_map": {}}

    for _, _, filenames in os.walk(via_json_dir):
        # sort ascending
        filenames_sorted = sorted(filenames)

        via_json_files_sorted = [f for f in filenames_sorted if os.path.splitext(f)[1] == ".json"]

        for i in range(len(via_json_files_sorted)):
            f = via_json_files_sorted[i]
            
            # acquire the video name
            if "_2." in f:
                video_filename, _ = f.split("_2")
            else:
                video_filename = os.path.splitext(f)[0]
            
            # add the video filename if we haven't before
            if video_filename not in video_file_id_map["filenames"]:
                video_file_id_map["filenames"].append(video_filename)

            # keeps track of the file ids that are associated with a particular video
            if video_filename not in video_file_id_map["id_map"]:
                video_file_id_map["id_map"][video_filename] = [i]
            else:
                video_file_id_map["id_map"][video_filename].append(i)


    with open(map_json_save_path, "w") as f:
        json.dump(video_file_id_map, f)


"""
====================================================================================================

    Helper function

====================================================================================================
"""

def printStdOutput(std_output):
    output_split = std_output.split('\n')
    for output in output_split:
        print(output)