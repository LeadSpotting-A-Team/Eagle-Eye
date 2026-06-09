from ultralytics import YOLO

# 1. טעינת המשקולות הקיימות שלך (המודל שכבר אומן)
# שנה את הנתיב למקום שבו נשמר ה-best.pt שלך
model = YOLO('runs/obb_yolov8s_dota/weights/best.pt')

# 2. אימון על ה-Dataset הראשון
model.train(data='data1.yaml', epochs=20, imgsz=1024, name='train_on_ds1')

# 3. אימון על ה-Dataset השני (המשקולות מתעדכנות מהשלב הקודם)
model.train(data='data2.yaml', epochs=20, imgsz=1024, name='train_on_ds2')

# 4. אימון על ה-Dataset השלישי (המשקולות מתעדכנות מהשלב הקודם)
model.train(data='data3.yaml', epochs=20, imgsz=1024, name='train_on_ds3')