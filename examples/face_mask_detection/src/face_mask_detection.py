import ATLearn
from ATLearn import task, algorithm

#%%
model = ATLearn.get_model(task.OBJECT_DETECTION,
                          algorithm.OD_STANDARD_TRANSFER,
                          data="../data/archive/train",
                          val_data="../data/archive/val",
                          network="yolov5s")
#%% train
model.train_model(model_file="../model_file/mask.pt")
#%% predict
model.predict(input_data="../../figs/face_mask.png", class_names=["w/o", "w/"],
              conf_thres=0.415, model_file="../model_file/mask.onnx")
