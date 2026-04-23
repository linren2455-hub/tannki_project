import sys
import cv2

# OpenPose読み込み
sys.path.append('/home/iwamoto/openpose/build/python/openpose')
import pyopenpose as op

# 設定
params = dict()
params["model_folder"] = "/home/iwamoto/openpose/models/"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame

    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 骨格付き画像
    output = datum.cvOutputData

    # 表示
    cv2.imshow("OpenPose", output)

    # 座標（1人目だけ）
    if datum.poseKeypoints is not None:
        person = datum.poseKeypoints[0]
        shoulder = person[2]
        elbow = person[3]
        wrist = person[4]

        print("肩:", shoulder, "肘:", elbow, "手首:", wrist)

    # qで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()