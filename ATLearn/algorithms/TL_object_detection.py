# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import ssl

import torch.jit

ssl._create_default_https_context = ssl._create_unverified_context
import random
import onnxruntime as ort
from tqdm import tqdm
from datetime import datetime
from typing import List
from ATLearn.algorithms.helper import *
from ATLearn.utils.yolo_data_loader import create_dataloader
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


class YoloModel(torch.nn.Module):
    def __init__(self, model):
        super(YoloModel, self).__init__()
        self.model = model

    def forward(self, img):
        # img0 = cv2.imread(input_data)
        # img = letterbox(img0, 640, stride=32, auto=False, scaleFill=True)[0]
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img)
        # img = img.unsqueeze(0)
        # img = img.float() / 255
        inputs, _ = self.model(img)
        inputs = non_max_suppression(inputs, conf_thres=0.4, iou_thres=0.45, classes=None,
                                     agnostic=False, max_det=1000)[0]
        return inputs


class TL_object_detection(object):
    def __init__(self, data=None, val_data=None, user_network=None, network='yolov5s', freeze=True, gpu_id=-1,
                 options=None, num_classes=2, epochs=100, batch_size=32, save_every_epoch=500):
        '''
        A standard object detection for object detection based on YOLOv5 or YOLOv3
        :param data: path to load the training examples, including both training images and annotations in one folder
        :param val_data: path to load the validation examples
        :param user_network: customers' own pre-trained model
        :param network: a large pre-trained YOLOv5 network
        :param freeze: whether to freeze the pre-trained layers
        :param gpu_id: whether to use GPUs
        :param num_classes: number of target classes
        :param epochs: total number of training epochs
        :param batch_size: batch size
        :param save_every_epoch: save checkpoint at some steps
        '''
        super(TL_object_detection, self).__init__()
        self.epochs = epochs
        self.save_every_epoch = save_every_epoch
        self.num_classes = num_classes
        self.save_traced_network = None
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        elif gpu_id >= 0 and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("No GPU is found! It will use CPU directly.")
        else:
            self.device = torch.device("cpu")

        if user_network is None:
            self.model = torch.hub.load('ultralytics/{}'.format(network[:6]),
                                        network,
                                        classes=num_classes,
                                        pretrained=True,
                                        autoshape=False,
                                        verbose=False).to(self.device)
        else:
            self.model = torch.load(user_network).to(self.device)

        self.base_names, self.detection_names = [], []
        num_freeze = 0
        if freeze:
            num_freeze = 10
        freeze_layers = [f'model.{x}.' for x in range(num_freeze)]
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            if any(x in name for x in freeze_layers):
                param.requires_grad = False
                self.base_names.append(name)
            else:
                self.detection_names.append(name)

        self.optimizer = smart_optimizer(self.model)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        gs = max(int(self.model.stride.max()), 32)
        imgsz = check_img_size(640, gs, floor=gs * 2)
        self.train_loader, self.dataset = create_dataloader(data, imgsz, batch_size, gs, shuffle=True)

        nl = self.model.model[-1].nl
        box = 0.02
        cls = 0.21638
        obj = 0.51728
        hyp = {}
        hyp['box'] = box * 3 / nl
        hyp['cls'] = cls * num_classes / 80 * 3 / nl
        hyp['obj'] = obj * (imgsz / 640) ** 2 * 3 / nl
        hyp['anchor_t'] = 3.3744
        self.model.hyp = hyp
        self.compute_loss = ComputeLoss(self.model)

        self.val_data = val_data
        if self.val_data:
            self.val_loader, _ = create_dataloader(self.val_data, imgsz, batch_size, gs)

    def train_model(self, model_file):
        t_start = time.time()
        for epoch in tqdm(range(1, self.epochs+1)):
            training_loss = 0.
            for imgs, targets, paths, _ in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                imgs = imgs.to(self.device).float() / 255

                gs = max(int(self.model.stride.max()), 32)
                imgsz = check_img_size(640, gs, floor=gs * 2)
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                pred = self.model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
                loss.backward()
                self.optimizer.step()
                training_loss += loss

            if self.val_data:
                print("Training epoch:", epoch)
                self.validation()

            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, training_loss)
        print("Model training is done with {:.4f} seconds!".format(time.time() - t_start))

        # jit save mask.pt
        self.model.eval()
        torch.save(self.model, model_file)

    def validation(self, iou_thres=0.2, conf_thres=0.001, save_txt=True):
        self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        niou = iouv.numel()
        seen = 0
        names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        jdict, stats, ap, ap_class = [], [], [], []
        for im, targets, paths, shapes in self.val_loader:
            im = im.to(self.device)
            targets = targets.to(self.device)
            im = im.float() / 255
            nb, _, height, width = im.shape
            out, _ = self.model(im)
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels
            out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=False)

            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                shape = shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=self.device)  # init
                seen += 1
                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                    continue

                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
        stats = [torch.cat(x, 0).detach().cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='../', names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=self.num_classes)  # number of targets per class
        else:
            nt = torch.zeros(1)
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    def predict(self,
                input_data: str,
                class_names: List[str],
                conf_thres=0.7,
                iou_thres=0.45,
                show_img=True,
                save_txt=True,
                model_file='./mask.pt'):
        is_file = Path(input_data).suffix[1:] in (IMG_FORMATS + VID_FORMATS) or input_data == "camera"
        assert is_file, f"Only image and video are supported now!"
        model_file_suffix = model_file.split('.')[-1].lower()
        if input_data.split('.')[-1].lower() in IMG_FORMATS:
            print("Image")
            img0 = cv2.imread(input_data)  # BGR
            assert img0 is not None, f'Image Not Found {input_data}'
            if 'pt' == model_file_suffix:
                img0 = self.show_results(img0, model_file, class_names, input_data, conf_thres, iou_thres, save_txt)
            elif 'onnx' == model_file_suffix:
                img0 = self.show_results_onnx(img0, model_file, class_names, input_data, conf_thres, iou_thres,
                                              save_txt)
            else:
                raise("model file suffix is not recoganized.")
            if show_img:
                cv2.imshow(str(Path(input_data)), img0)
                cv2.waitKey(0)
        elif input_data.split('.')[-1].lower() in VID_FORMATS or input_data == "camera":
            print("Video")
            if input_data == "camera":
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(input_data)
            ret_val, img0 = cap.read()
            while ret_val:
                _, img0 = cap.read()
                if 'pt' == model_file_suffix:
                    img0 = self.show_results(img0, model_file, class_names, input_data, conf_thres, iou_thres, save_txt)
                elif 'onnx' == model_file_suffix:
                    img0 = self.show_results_onnx(img0, model_file, class_names, input_data, conf_thres, iou_thres,
                                                  save_txt)
                else:
                    raise ("model file suffix is not recoganized.")
                if show_img:
                    cv2.imshow(str(Path(input_data)), img0)
                    cv2.waitKey(1)
                ret_val, img0 = cap.read()
        else:
            raise("Unknown input format! Only image and video are supported now!")

    def show_results(self, img0, model_file, class_names, input_data, conf_thres, iou_thres, save_txt):
        self.model = torch.load(model_file)
        self.model.eval()

        img = letterbox(img0, 640, stride=32, auto=False, scaleFill=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.unsqueeze(0)
        img = img.float() / 255
        pred, _ = self.model(img)

        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None,
                                   agnostic=False, max_det=1000)[0]

        colors = Colors()
        annotator = Annotator(img0, line_width=3, example=str(class_names))
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for c in pred[:, -1].unique():
            n = (pred[:, -1] == c).sum()  # detections per class
            s = f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(s)
        lines = []
        for *xyxy, conf, cls in reversed(pred):
            c = int(cls)  # integer class
            label = f'{class_names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf)
            lines.append(line)

        if save_txt:
            save_path = input_data.rsplit('.', 1)[0] + '.txt'
            with open(save_path, 'a') as f:
                for line in lines:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        img0 = annotator.result()

        return img0

    def show_results_onnx(self, img0, model_file, class_names, input_data, conf_thres, iou_thres, save_txt):
        # Load the ONNX model
        # model_file = "./onnx_models_100/mask.onnx"
        session = ort.InferenceSession(model_file)

        # Define input and output names
        input_name = session.get_inputs()[0].name
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]

        img = letterbox(img0, 640, stride=32, auto=False, scaleFill=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0

        # Run inference
        input_data_onnx = {input_name: img}
        pred = session.run(output_names, input_data_onnx)[0]
        pred = torch.from_numpy(pred).to(self.device)

        # Post-process
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None,
                                   agnostic=False, max_det=1000)[0]

        # Visualization
        colors = Colors()
        annotator = Annotator(img0, line_width=3, example=str(class_names))
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for c in pred[:, -1].unique():
            n = (pred[:, -1] == c).sum()  # detections per class
            s = f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(s)
        lines = []
        for *xyxy, conf, cls in reversed(pred):
            c = int(cls)  # integer class
            label = f'{class_names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf)
            lines.append(line)

        if save_txt:
            save_path = input_data.rsplit('.', 1)[0] + '.txt'
            with open(save_path, 'a') as f:
                for line in lines:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        img0 = annotator.result()

        return img0

    def export(self, save_name="model", save_path="../save/"):
        im = torch.rand(1, 3, 640, 640).to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        for _ in range(1):
            y = self.model(im)
        self.save_traced_network = torch.jit.trace(self.model, im)
        # torch.save(self.save_traced_network, '{}/{}.pt'.format(save_path, save_name))
        self.save_traced_network.save('{}/{}.pt'.format(save_path, save_name))
        torch.save({
            'base_params_names': self.base_names,
            'detection_params_names': self.detection_names
        }, '{}/{}_params.pt'.format(save_path, save_name))
        self.print_text("TL model has been traced, and saved at {}/{}.pt".format(save_path, save_name))

    def save_checkpoint(self, epoch, training_loss):
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.model.eval()
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': training_loss,
        }, "../checkpoint/checkpoint_epoch_{}_{}.pt".format(epoch, date))

    @staticmethod
    def print_text(text):
        print("\033[91m {}\033[00m".format(text))

