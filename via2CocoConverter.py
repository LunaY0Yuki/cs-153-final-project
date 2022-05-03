import numpy as np
from datetime import date
import os
import json
import ffmpeg
from math import ceil
import traceback
from merge_coco.merge import combine


class CocoIdGenerator:
    """
    Coco Id Generator generates the correct annotation id and image id in the following format:
    
    If digits_for_file=3, digits_for_frame=5, digits_for_obj=3, 
    then the id would be formatted as the following:
        _ _ _ _ _ _ _ _ _ _ _

    where if we break it down into sections:
        _ _ _   _ _ _ _ _ _   _ _ _
       {  1  } {     2     } {  3  }

    and each section is semantically defined as
        1 = the file id, unique for each VIA annotation
        2 = the frame id, unique for each frame in the video (in units of 0.1 sec)
        3 = the object id, unique for each identified object in a video


    Parameters:
        file_id - id, the video id for this particular via annotation
        digits_for_file - int, default = 3
            For the file id in the generated annotation id and image id, it gets allocated {digits_for_file} digits
        digits_for_frame - int, default = 5
            For the frame id in the generated annotation id and image id, it gets allocated {digits_for_frame} digits
        digits_for_obj - int, default = 3
            For the object id in the  generated annotation id and image id, it gets allocated {digits_for_obj} digits
    """
    def __init__(self, file_id, digits_for_file=3, digits_for_frame=5, digits_for_obj=3):
        self.file_id = file_id
        self.digits_for_file = digits_for_file
        self.digits_for_frame = digits_for_frame
        self.digits_for_obj = digits_for_obj

    def generateImageId(self, curr_time_int):
        file_sec = str(self.file_id)
        file_sec = file_sec.zfill(self.digits_for_file)
        
        image_sec = str(curr_time_int)
        image_sec = image_sec.zfill(self.digits_for_frame)

        return int(file_sec + image_sec)

    def generateAnnId(self, curr_time_int, obj_id):
        file_sec = str(self.file_id)
        file_sec = file_sec.zfill(self.digits_for_file)
        
        image_sec = str(curr_time_int)
        image_sec = image_sec.zfill(self.digits_for_frame)
        
        obj_sec = str(obj_id)
        obj_sec = obj_sec.zfill(self.digits_for_obj)

        return int(file_sec + image_sec + obj_sec)


def getLabelAndId(attrDict, attrConfigDict):
    label = None
    object_id = None

    object_present_exist = False
    object_present_attr_key = ""
    object_id_exist = False
    object_id_attr_key = ""
    object_label_exist = False
    object_label_attr_key = ""

    mapToStandardLabel = {}

    for k in attrDict:
      if attrConfigDict[k]["aname"] == "object_present":
          object_present_exist = True
          object_present_attr_key = k
      elif attrConfigDict[k]["aname"] == "object_id":
          object_id_exist = True
          object_id_attr_key = k
      elif attrConfigDict[k]["aname"] == "object_label":
          object_label_exist = True
          object_label_attr_key = k
          
          cat_options = attrConfigDict[k]["options"]
          for og_cat_id in cat_options:
            if "shark" in cat_options[og_cat_id] or "0" == cat_options[og_cat_id]:
              mapToStandardLabel[int(og_cat_id)] = 1
            elif "human" in cat_options[og_cat_id] or "1" == cat_options[og_cat_id]:
              mapToStandardLabel[int(og_cat_id)] = 2
            else:
              print(f"not recognizable options in cat_options: {cat_options}")
              print("error at getLabelAndId")
      else:
          print(f"attrDict key: {k}")
          print("error at getLabelAndId")

    # default map
    #   shark = 0 -> 1
    #   human = 1 -> 2
    if len(mapToStandardLabel):
      mapToStandardLabel = {0:1, 1:2}

    if object_label_exist:
        original_label = int(attrDict[object_label_attr_key])
        label = mapToStandardLabel[original_label]
    elif object_present_exist:
        if "shark" in attrDict[object_present_attr_key] or "0" == attrDict[object_present_attr_key]:
            label = 1  # coco's 0 category id is 
        elif "human" in attrDict[object_present_attr_key] or "1" == attrDict[object_present_attr_key]:
            label = 2
        else:
            print(f"cannot identify object id based on: { attrDict[object_present_attr_key]}")
    else:
        print("cannot use any attribute to find label")
        print("error at getLabelAndId")

    if object_id_exist:
      object_id = int(attrDict[object_id_attr_key])

    return label, object_id

def getSegmentation(xy_region):
  _, x, y, w, h = xy_region
  return [[x, y, x+w, y, x+w, y+h, x, y+h]]

def getBBox(xy_region):
  _, x, y, w, h = xy_region
  return [x, y, w, h]

def getArea(xy_region):
  _, _, _, w, h = xy_region
  return w * h

def getCurrObjId(curr_time, cat_id, og_obj_id, curr_obj_id_dict):
  curr_time_int = int(curr_time * 10)

  if og_obj_id != None and (og_obj_id in curr_obj_id_dict[curr_time_int]["existing_obj_ids"][cat_id]):
    print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) already got added, so got ignored")
    return None, curr_obj_id_dict
  else:
    curr_obj_id_dict[curr_time_int]["existing_obj_ids"][cat_id] += [og_obj_id]
    curr_obj_id = curr_obj_id_dict[curr_time_int]["curr_obj_id"]
    curr_obj_id_dict[curr_time_int]["curr_obj_id"] += 1
    return curr_obj_id, curr_obj_id_dict

ANN_AREA_FILTER_THRESHOLD = 5

# https://stackoverflow.com/questions/42021972/truncating-decimal-digits-numpy-array-of-floats
def trunc(values, decs=1):
    return np.trunc(values*10**decs)/(10**decs)

"""
- get the time stamp so that we can create the annotation id (how to deal with the object annoation part)
- have a dictionary (we could initalize it): for each time stamp (key), we keep track of the curr object annotation id that we should use
- steal; "area": float, "bbox": [x,y,width,height],
- iscrowd = 0
- errorhandling
"""
def createCocoAnnotationDict(viaObjAnnotation, viaCatConfig, higest_z, idGenerator):
  curr_obj_id_dict = {}
  for z in range(0, ceil(higest_z * 10)):
    curr_obj_id_dict[z] = {"curr_obj_id": 0, "existing_obj_ids": {1: [], 2:[]}}

  cocoAnnotations = []

  for k in viaObjAnnotation:
      ann = viaObjAnnotation[k]
      # if len of z is more than 1, not a bounding box annotation
      if len(ann["z"]) == 1:
          curr_time = ann["z"][0]
          curr_time_int = int(ann["z"][0] * 10)

          cat_id, og_obj_id = getLabelAndId(ann["av"], viaCatConfig)
          
          if ann["xy"][0] != 2:
            print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) doesn't have right format for xy labels, only expect 4 points for bounding box")
          else:
            area = getArea(ann["xy"])
            if area < ANN_AREA_FILTER_THRESHOLD:
              print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) has unreasonably small area: {area}")
            elif curr_time > higest_z:
              print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) exceeds video length, so gets ignored")
            else:
              curr_obj_id, curr_obj_id_dict = getCurrObjId(curr_time, cat_id, og_obj_id, curr_obj_id_dict)
              
              if curr_obj_id != None:
                cocoAnnotations.append({
                                        "id": idGenerator.generateAnnId(curr_time_int, curr_obj_id), 
                                        "image_id": idGenerator.generateImageId(curr_time_int), 
                                        "category_id": cat_id, 
                                        "segmentation": getSegmentation(ann["xy"]), 
                                        "area": area, 
                                        "bbox": getBBox(ann["xy"]), 
                                        "iscrowd": 0,
                                      })

  return cocoAnnotations


def createCocoImageDict(w, h, higest_z, video_filename, idGenerator):
  cocoImages = []

  for z in range(0, ceil(higest_z * 10)):
    image_num_in_filename = str(z)
    image_num_in_filename = image_num_in_filename.zfill(idGenerator.digits_for_image)

    image_filename = f"{video_filename}_" + image_num_in_filename + ".jpg"
    
    cocoImages.append({
                        "id": idGenerator.generateImageId(z), 
                        "width": w, 
                        "height": h, 
                        "file_name": image_filename, 
                        "license": 0, 
                        "flickr_url": "", 
                        "coco_url": "", 
                        "date_captured": date.today().strftime("%m/%d/%Y"),
    })

  return cocoImages


def createCocoInfoDict(video_filename):
  return {
          "year": date.today().year, 
          "version": "", 
          "description": video_filename, 
          "contributor": "", 
          "url": "", 
          "date_created": date.today().strftime("%m/%d/%Y"),
  }

# from the original conveter
def createCocoLisenses():
  return [{
            "id": 0,
            "name": "Unknown License",
            "url": ""
        }]

def createCocoCategories():
  return [{
            "supercategory": "object_label",
            "id": 1,
            "name": "shark"
          },
          {
            "supercategory": "object_label",
            "id": 2,
            "name": "human"
          }]

def getFilenameWithoutPath(file_path):
  path_without_extension = os.path.splitext(file_path)[0]

  return path_without_extension.split("/")[-1]

def convertToCocoFormat(via_json_path, video_dir, coco_json_dir, video_id):
  with open(via_json_path, 'r') as f:
    via_json = json.load(f)

  via_json_name = getFilenameWithoutPath(via_json_path)
  print(f"============ video id = {video_id}, via json name = {via_json_name} ============")
  print()

  # assume that there is only one filename
  video_filename = via_json["file"]["1"]["fname"]
  video_path = video_dir + video_filename
  
  vid_info = ffmpeg.probe(video_path)

  # https://stackoverflow.com/questions/7362130/getting-video-dimension-resolution-width-x-height-from-ffmpeg
  height = int(vid_info['streams'][0]['height'])
  width = int(vid_info['streams'][0]['width'])
  print(f"h = {height}, w = {width}")

  # https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
  vid_length = float(vid_info['format']['duration'])
  print(f"vid length (in sec): {vid_length}")
  print()

  coco_json_save_path = coco_json_dir + via_json_name + "_coco.json"

  idGen = CocoIdGenerator(file_id = video_id)

  coco_json = {
                  "info": createCocoInfoDict(via_json_name), 
                  "images": createCocoImageDict(width, height, vid_length, via_json_name, idGen), 
                  "annotations": createCocoAnnotationDict(via_json["metadata"], via_json["attribute"], vid_length, idGen), 
                  "categories": createCocoCategories(),
                  "licenses": createCocoLisenses(),
                }
  
  print()

  with open(coco_json_save_path, 'w') as f:
    json.dump(coco_json, f)


def mergeAllCoco(coco_json_dir, merged_save_path):
    coco_json_files = []

    for dirpath, _, filenames in os.walk(coco_json_dir):
        for f in filenames:
            if os.path.splitext(f)[1] == ".json":
                coco_json_files.append(os.path.join(dirpath, f))
    
    merged_coco_file = coco_json_files[0]

    coco_json_with_errors = []

    for i in range(1, len(coco_json_files)):
        try: 
            coco_json_path = coco_json_files[i]

            if i != 1:
                merged_coco_file = merged_save_path
                
            combine(merged_coco_file, coco_json_path, merged_save_path)
        except:
            trace_error = traceback.format_exc()
            
            print(trace_error)

            print("xxxxxxxxxxx WARNING! xxxxxxxxxxx")
            print(f"xxxxxxx filename = {coco_json_path} xxxxxxx")
            print()
            
            coco_json_with_errors.append((coco_json_path, trace_error))

            pass

    
    print("------------ List of COCO Files that Have Errors When Converting ------------")
    print("     format: (coco json filename, error message)")
    for p in coco_json_with_errors:
        fn, em = p
        print(fn)
        print(em)
        print()
            def mergeAllCoco(coco_json_dir, merged_save_path):
    coco_json_files = []

    for dirpath, _, filenames in os.walk(coco_json_dir):
        for f in filenames:
            if os.path.splitext(f)[1] == ".json":
                coco_json_files.append(os.path.join(dirpath, f))
    
    merged_coco_file = coco_json_files[0]

    coco_json_with_errors = []

    for i in range(1, len(coco_json_files)):
        try: 
            coco_json_path = coco_json_files[i]

            if i != 1:
                merged_coco_file = merged_save_path
                
            combine(merged_coco_file, coco_json_path, merged_save_path)
        except:
            trace_error = traceback.format_exc()
            
            print(trace_error)

            print("xxxxxxxxxxx WARNING! xxxxxxxxxxx")
            print(f"xxxxxxx filename = {coco_json_path} xxxxxxx")
            print()
            
            coco_json_with_errors.append((coco_json_path, trace_error))

            pass

    
    print("------------ List of COCO Files that Have Errors When Converting ------------")
    print("     format: (coco json filename, error message)")
    for p in coco_json_with_errors:
        fn, em = p
        print(fn)
        print(em)
        print()
            