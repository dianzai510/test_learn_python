from typing import Any
import cv2
import numpy as np
import onnxruntime as ort
import torch
from our1314.myutils.myutils import sigmoid

class yolo:
    def __init__(self, onnx_path, confidence_thres=0.1, iou_thres=0):
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_name = model_inputs[0].name
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
    
    def __call__(self, img):
        image_data = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: image_data})
        output_img = self.postprocess(img, outputs)
        return output_img

    #预处理
    def preprocess(self, input_image):
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #region pad
        img_height, img_width = input_image.shape[:2]
        scale = 640.0 / max(img_height,img_width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        self.up, self.left = (640-img.shape[0])//2, (640-img.shape[1])//2
        down = 640 - img.shape[0] - self.up
        right = 640 - img.shape[1] - self.left
        #endregion

        img = cv2.copyMakeBorder(img, self.up, down, self.left, right, cv2.BORDER_CONSTANT, value=(128,128,128))

        image_data = np.array(img) / 255.0
        #image_data = (image_data - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])

        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data
    
    #后处理
    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))
        proto = output[1]
        print(outputs[0])
        
        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        img_height, img_width = input_image.shape[:2]
        
        scale = 640.0/(max(img_height, img_width))
        index = []
        for i in range(rows):
            rrr = outputs[i]
            classes_scores = outputs[i][4:5]
            max_score = np.amax(classes_scores)
            
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = (x-self.left-w/2)//scale
                top = (y-self.up-h/2)//scale
                width = w//scale
                height = h//scale

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                index.append(i)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)


        pass
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

            ##########################################
            
            idx = index[i]
            r = outputs[idx]
            masks_in = r[5:]
            masks_in = masks_in[None,]
            c,mh,mw = proto[0].shape
            aaa = proto[0].reshape(32,-1)
            bbb = masks_in @ aaa
            ccc = sigmoid(bbb)
            ddd = ccc.reshape(-1,mh,mw)

            box = r[:4]
            box = np.array(box)
            box = box[None,]
            masks = self.process_mask(proto[0], masks_in, box, [640,640], True)
            masks = masks.astype(np.uint8)
            masks = np.transpose(masks, (1,2,0))
            cv2.imshow("dis", masks*255)
            cv2.moveWindow("dis",0,0)
            cv2.waitKey()
            pass

        # Return the modified input image
        return input_image

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0,255,255), 2)
        # label = 'a'
        # (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # label_x = x1
        # label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), (0,255,255), cv2.FILLED)
        # cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape

        masks = (masks_in @ protos.reshape(32,-1))
        masks = sigmoid(masks)
        masks = masks.reshape(-1, mh, mw)
        
        downsampled_bboxes = self.wywh2xyxy(bboxes)
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        a = masks > 0.5
        
        
        # if upsample:
        #     masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        return a
    
    def crop_mask(self, masks, boxes):
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        x1 = x1[None,None,]
        y1 = y1[None,None,]
        x2 = x2[None,None,]
        y2 = y2[None,None,]

        n, h, w = masks.shape
        #x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.arange(w)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def wywh2xyxy(self, box_xywh):
        box_xyxy = box_xywh.copy()
        box_xyxy[:, 0] = box_xywh[:,0]-box_xywh[:,2]/2
        box_xyxy[:, 1] = box_xywh[:,1]-box_xywh[:,3]/2
        box_xyxy[:, 2] = box_xywh[:,0]+box_xywh[:,2]/2
        box_xyxy[:, 3] = box_xywh[:,1]+box_xywh[:,3]/2
        return box_xyxy

if __name__ == "__main__":
    seg = yolo('yolov8-seg.onnx')
    src = cv2.imdecode(np.fromfile('D:/desktop/choujianji/out1/train/2023-07-20_16.54.02-278_back (4).jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
    dis = seg(src)
    cv2.imshow("dis", dis)
    cv2.waitKey()