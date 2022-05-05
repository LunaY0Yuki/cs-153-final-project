import argparse

from via2CocoConverter import convertAllViaToCoco, mergeAllCoco
from video2FrameConverter import convertAllVideosToFrames, generatetVidToFileIdMap

"""
Overall main function to execute the entire workflow of our data processing pipeline
    Will save the log of any error in the log file directory specified in config.json
"""
def main(via_json_dir, video_dir, coco_json_dir, 
            merged_coco_json_path, 
            video_frame_dir, 
            map_json_save_path):
    print("************************************************")
    print()
    print("     Converting ALL via jsons to coco jsons")
    print()
    print("************************************************\n")

    convertAllViaToCoco(via_json_dir, video_dir, coco_json_dir)

    print("\n\n************************************************")
    print()
    print("     Merging ALL coco jsons to ONE coco json")
    print()
    print("************************************************\n")

    mergeAllCoco(coco_json_dir, merged_coco_json_path)

    print("\n\n************************************************")
    print()
    print("     Convert all videos to frames")
    print()
    print("************************************************\n")  

    convertAllVideosToFrames(via_json_dir, video_dir, video_frame_dir) 

    print("\n\n************************************************")
    print()
    print("     Create the video filename to file id map json")
    print()
    print("************************************************\n")  

    generatetVidToFileIdMap(via_json_dir, map_json_save_path)


if __name__ == '__main__':
    """
    Note:
        all the path to directory should have / at the end
        all the directories should be created already

    Example shell command: 

        python3 main.py -v "./via_annotations/" -d "./videos/" -c "./coco_annotations/" -m "/merged_coco_annotation/merged_coco.json" -f "./frames/" -a "./video_file_id_map.json"

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--via", type=str, help="path to the via jsons' directory", required=True)
    parser.add_argument("-d", "--video", type=str, help="path to the videos's directory", required=True)
    parser.add_argument("-c", "--coco", type=str, help="path of where to save individual coco's file", required=True)
    parser.add_argument("-m", "--mergedcoco", type=str, help="path (include filename w/ json) to save the merged coco json", required=True)
    parser.add_argument("-f", "--frame", type=str, help="path of where to save the video frames", required=True)
    parser.add_argument("-a", "--map", type=str, help="path (include filename w/ .json) to save the map from video filename to file id", required=True)
    
    args = parser.parse_args()

    main(via_json_dir = args.via, video_dir = args.video, coco_json_dir = args.coco, 
            merged_coco_json_path = args.mergedcoco, 
            video_frame_dir = args.frame, 
            map_json_save_path = args.map)

