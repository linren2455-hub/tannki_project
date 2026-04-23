import sys
import cv2

sys.path.append('/home/iwamoto/openpose/build/python/openpose')
import pyopenpose as op

params = dict()
params["model_folder"] = "/home/iwamoto/openpose/models/"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

image = cv2.imread("/home/iwamoto/openpose/examples/media/COCO_val2014_000000000192.jpg")

datum = op.Datum()
datum.cvInputData = image

opWrapper.emplaceAndPop(op.VectorDatum([datum]))

print(datum.poseKeypoints)
print(datum.poseKeypoints.shape)