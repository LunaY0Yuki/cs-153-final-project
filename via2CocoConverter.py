import numpy as np
from datetime import date
import os
import json
import ffmpeg
from math import ceil
import traceback
from merge_coco.merge import combine

ANN_AREA_FILTER_THRESHOLD = 5

"""
====================================================================================================

    Main functions to run

====================================================================================================
"""

"""
Convert all the via annotations in {via_json_dir} to coco annotations, which get stored in {coco_json_dir}

Warning:
  All the directory path should end with "/"

Parameters:
  via_json_dir - string, path to the directory that contains all the via_annotations
    WARNING:
      - we assume that the via annotation jsons have the same filename as the corresponding vidoe filename
      - if there exist another via annotation jsons for the same video, 
          the via annotation json would just have "_2" at the end
  video_dir - string, path to the directory that contains all the videos
  coco_json_dir - string, path to the directory where we would save the coco annotation jsons
"""
def convertAllViaToCoco(via_json_dir, video_dir, coco_json_dir):
    via_json_files = []

    for dirpath, _, filenames in os.walk(via_json_dir):
        # sort ascending order
        #   Warning: order is essentially the video id / file id
        filenames_sorted = sorted(filenames)

        for f in filenames_sorted:
            if os.path.splitext(f)[1] == ".json":
                via_json_files.append(os.path.join(dirpath, f))
    
    # keeps track of via annotations that have problem during conversion
    via_json_with_errors = []

    for i in range(len(via_json_files)):
        via_json_file = via_json_files[i]

        try: 
            convertToCocoFormat(via_json_file, video_dir, coco_json_dir, file_id = i)
        except:
            trace_error = traceback.format_exc()
            
            print(trace_error)

            print("xxxxxxxxxxx WARNING! xxxxxxxxxxx")
            print(f"xxxxxxx video id = {i}, via json name = {via_json_file} xxxxxxx")
            print()
            
            via_json_with_errors.append((via_json_file, i, trace_error))

            pass

    
    if via_json_with_errors != []:
        print("------------ List of VIA Files that Have Errors When Converting ------------")
        print("     format: (json file path, video id that should be used)")

        for fpath, file_id, trace_error in via_json_with_errors:
            print((fpath, file_id))
            print(trace_error)
            print()


"""
Convert a single via annotation, specified by {via_json_path}, to coco annotations, 
  and store the converted coco annotation in coco_json_dir

Warning:
  All the directory path should end with "/"

Parameters:
  via_json_dir - string, path to the directory that contains all the via_annotations
    WARNING:
      - we assume that the via annotation jsons have the same filename as the corresponding vidoe filename
      - if there exist another via annotation jsons for the same video, 
          the via annotation json would just have "_2" at the end
  video_dir - string, path to the directory that contains all the videos
  coco_json_dir - string, path to the directory where we would save the coco annotation jsons
  file_id - int, unique identifier for a particular via annotation
"""
def convertToCocoFormat(via_json_path, video_dir, coco_json_dir, file_id):
  with open(via_json_path, 'r') as f:
    via_json = json.load(f)

  via_json_name = getFilenameWithoutPath(via_json_path)

  print(f"============ video id = {file_id}, via json name = {via_json_name} ============")
  print()

  # assume that there is only one filename in the via annotation 
  video_filename = via_json["file"]["1"]["fname"]
  video_path = video_dir + video_filename
  
  vid_info = ffmpeg.probe(video_path)

  # Source: https://stackoverflow.com/questions/7362130/getting-video-dimension-resolution-width-x-height-from-ffmpeg
  height = int(vid_info['streams'][0]['height'])
  width = int(vid_info['streams'][0]['width'])
  print(f"h = {height}, w = {width}")

  # Source: https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
  vid_length = float(vid_info['format']['duration'])
  print(f"vid length (in sec): {vid_length}")
  print()

  coco_json_save_path = coco_json_dir + via_json_name + "_coco.json"

  # create the annotation id and image id generator for this conversion
  idGen = CocoIdGenerator(file_id = file_id)

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


"""
====================================================================================================

    Annotation id and image id generator for converting VIA annotations to Coco json

====================================================================================================
"""
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


    """
    Before the image id gets converted to integer (will lose any leading 0s), the image id has the format:
        _ ... _   _ _ ... _ _  
       {   1   } {     2     }

    and each section is semantically defined as
        1 = the file id, unique for each VIA annotation, has self.digits_for_file number of digits
        2 = the frame id, unique for each frame in the video (in units of 0.1 sec), has self.digits_for_frame number of digits

    Parameters:
        curr_time_int - int, the current time stamp in unit in 0.1 second

    Return:
        image id as an interger (losing the leading 0s)
    """
    def generateImageId(self, curr_time_int):
        file_sec = str(self.file_id)
        file_sec = file_sec.zfill(self.digits_for_file)
        
        image_sec = str(curr_time_int)
        image_sec = image_sec.zfill(self.digits_for_frame)

        return int(file_sec + image_sec)


    """
    Before the annotation id gets converted to integer (will lose any leading 0s), the annotation id has the format:
        _ ... _   _ _ ... _ _   _ ... _
       {   1   } {     2     } {   3   }

    and each section is semantically defined as
        1 = the file id, unique for each VIA annotation, has self.digits_for_file number of digits
        2 = the frame id, unique for each frame in the video (in units of 0.1 sec), has self.digits_for_frame number of digits
        3 = the object id, unique for each identified object in a video, has self.digits_for_object number of digits

    Parameters:
        curr_time_int - int, the current time stamp in unit in 0.1 second
        obj_id - int, unique identifier for the object that appears at a particular time stamp

    Return:
        annotation id as an interger (losing the leading 0s)
    """
    def generateAnnId(self, curr_time_int, obj_id):
        file_sec = str(self.file_id)
        file_sec = file_sec.zfill(self.digits_for_file)
        
        image_sec = str(curr_time_int)
        image_sec = image_sec.zfill(self.digits_for_frame)
        
        obj_sec = str(obj_id)
        obj_sec = obj_sec.zfill(self.digits_for_obj)

        return int(file_sec + image_sec + obj_sec)


"""
====================================================================================================

    Main helper fucntions called by convertToCocoFormat function
      - createCocoInfoDict
      - createCocoImageDict
      - createCocoAnnotationDict
      - createCocoCategories
      - createCocoLisenses

====================================================================================================
"""

"""
Parameter:
  video_filename - string, filename of the via json without the extension
"""
def createCocoInfoDict(video_filename):
  return {
          "year": date.today().year, 
          "version": "", 
          "description": video_filename, 
          "contributor": "", 
          "url": "", 
          "date_created": date.today().strftime("%m/%d/%Y"),
  }


"""
Generate a list of images in Coco format that appear in this annotatations

Parameter:
  w - int, width of the frame in pixel
  h - int, height of the frame in pixel
  highest_z - float, video length in seconds
  video_filename - string, the frame filename will be defined as {video_filename}_{frame id}.jpg
    video_filename here would be the same as via annotation's filename
  idGenerator - CocoIdGenerator object, helps generate image id
"""
def createCocoImageDict(w, h, highest_z, video_filename, idGenerator):
  cocoImages = []

  # iterate in each frame
  #   notice that z is an integer (represents time in 0.1 seconds)
  for z in range(0, ceil(highest_z * 10)):
    # z is essentially the frame number
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


"""
- get the time stamp so that we can create the annotation id (how to deal with the object annoation part)
- have a dictionary (we could initalize it): for each time stamp (key), we keep track of the curr object annotation id that we should use
- steal; "area": float, "bbox": [x,y,width,height],
- iscrowd = 0
- errorhandling

Generate a list of object annotation in Coco format based on the original via annotation

Meanwhile, also filter and clean the original via annotation based on the following rules.
  We ignore an annotation: 
    - if the bounding box is less than ANN_AREA_FILTER_THRESHOLD
    - if its associated timestamp is higher than the video length
    - if an annotation with the exact same category and object id already exists

Parameters:
  viaObjAnnotation - dictionary, via annotation for all the objects in the video
    from via annotation's "metadata"
  viaCatConfig - dictionary, via category information for the annotation json
    from via annotation's "attribute"
  highest_z - float, video length in seconds
  idGenerator - CocoIdGenerator object, helps generate annotation id and image id
"""
def createCocoAnnotationDict(viaObjAnnotation, viaCatConfig, highest_z, idGenerator):
  # keeps track of the via object id that we have seen so far
  #   and also the current object id that should be used in idGenerator
  curr_obj_id_dict = {}
  for z in range(0, ceil(highest_z * 10)):
    curr_obj_id_dict[z] = {"curr_obj_id": 0, "existing_obj_ids": {1: [], 2:[]}}

  cocoAnnotations = []

  for k in viaObjAnnotation:
      ann = viaObjAnnotation[k]

      # if len of z is more than 1, not a bounding box annotation
      if len(ann["z"]) == 1:
          curr_time = ann["z"][0]  # in seconds
          curr_time_int = int(ann["z"][0] * 10)  # int, representing in unit 0.1 second

          # get the category id and the original object id
          cat_id, og_obj_id = getLabelAndId(ann["av"], viaCatConfig)
          
          if ann["xy"][0] != 2:
            print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) doesn't have right format for xy labels, only expect 4 points for bounding box")
          else:
            area = getArea(ann["xy"])

            if area < ANN_AREA_FILTER_THRESHOLD:
              print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) has unreasonably small area: {area}")
            elif curr_time > highest_z:
              print(f"Ann (cat_id={cat_id}, og_obj_id={og_obj_id}, t={curr_time}) exceeds video length, so gets ignored")
            else:
              # get the actual object id that will be used by the idGenerator
              curr_obj_id, curr_obj_id_dict = getCurrObjId(curr_time, cat_id, og_obj_id, curr_obj_id_dict)
              
              if curr_obj_id != None:
                # iscrowd = 0 means that the ann is not used to label large groups of objects (e.g. a crowd of people).
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


"""
Create information about the categories in coco format

  object_label
    1 - shark
    2 - human

Notes:
  we cannot use 0 because 0 is reserved as a special class for Coco 
"""
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


"""
Create information about the lisences in coco format (does not actually get used)

 (return the same thing the via provided via2coco converter)
"""
def createCocoLisenses():
  return [{
            "id": 0,
            "name": "Unknown License",
            "url": ""
        }]


"""
====================================================================================================

    Other helper function used

====================================================================================================
"""

"""
Based on the attribute dictionary {attrDict} for a particular object attribute 
"""
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


# https://stackoverflow.com/questions/42021972/truncating-decimal-digits-numpy-array-of-floats
def trunc(values, decs=1):
    return np.trunc(values*10**decs)/(10**decs)





def getFilenameWithoutPath(file_path):
  path_without_extension = os.path.splitext(file_path)[0]

  return path_without_extension.split("/")[-1]


            

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
