"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
from random import random

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

roi_boxes = []
roi_classes = []

def segment(img, roi_boxes=None, roi_classes=None):
    # initial black image
    result_img = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3)).astype(np.uint8)
    print(f"Rois: {len(roi_boxes)}")
    with_splash = False

    if len(roi_boxes) > 1:
        with_splash = True


    if len(roi_boxes) < 1:
        print("roi null")
        return result_img

    for roi, roi_class in zip(roi_boxes, roi_classes):
        if roi_class == 1:
            img_seg = segment_splash(img, roi)
            #cv2.imshow("segment_splash", img_seg)
        if roi_class == 0:
            img_seg = segment_diver(img, roi, with_splash)
            #cv2.imshow("segment_diver", img_seg)

        result_img = cv2.add(img_seg, result_img)
        #cv2.imshow("merged result", result_img)


    return result_img



def segment_splash(splash_img, roi_box):

    # generate black image
    black_image = np.zeros((np.array(splash_img).shape[0], np.array(splash_img).shape[1], 3)).astype(np.uint8)

    # set roi
    x = roi_box

    # get roi coordinates
    left, top = int(x[0]), int(x[1])
    left_w, top_h = int(x[2]), int(x[3])
    # cut roi
    roi = splash_img[top:top_h, left:left_w]

    # initialize new Image
    """
    generate new image
    """
    # 3 channel roi
    img_3c_roi = np.zeros((np.array(roi).shape[0], np.array(roi).shape[1], 3)).astype(np.uint8)

    # 3 channel full image
    img_3c_full = np.zeros((np.array(splash_img).shape[0], np.array(splash_img).shape[1], 3)).astype(np.uint8)

    """
    segmentation
    """
    # median blur
    splash_img_blr = cv2.medianBlur(splash_img,3)
    #cv2.imshow("1: BLur", splash_img_blr)
    # convert to gray
    full_image_gray = cv2.cvtColor(splash_img_blr, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("2: Grayscale", full_image_gray)
    # stretch contrast histeq
    img_equ = cv2.equalizeHist(full_image_gray)
    #cv2.imshow("3: Equalize",img_equ)


    # threshold image
    thresh, splash_thresh_full_image = cv2.threshold(img_equ, 250, 255, cv2.THRESH_BINARY)
    #cv2.imshow("4: Threshold", splash_thresh_full_image)
    # morphological operations
    # set kernel for erosion
    kernel_splash = np.ones((1, 1), np.uint8)

    # apply erosion
    erosion_full_image = cv2.erode(splash_thresh_full_image, kernel_splash, iterations=1)
    #cv2.imshow("5: Erosion", erosion_full_image)



    img_3c_full[:, :, 0] = erosion_full_image
    img_3c_full[:, :, 1] = erosion_full_image
    img_3c_full[:, :, 2] = erosion_full_image

    # find contours
    contours_1, hier_1 = cv2.findContours(erosion_full_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_1) > 1:
        # find contours
        contours_1 = sorted(contours_1, key=cv2.contourArea)
        # roi
        #cv2.rectangle(img_3c_full,(left, top),(left_w, top_h),(0, 0, 255), 3)

        for c in contours_1:
            #cv2.drawContours(img_3c_full, small_c, -1, (0,255,0), 3)
            # fill contours
            cv2.fillPoly(img_3c_full, pts=[c], color=(0,0,0))

        # splash contour
        #cv2.rectangle(img_3c_full, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 255, 0), 3)
        max_con = max(contours_1, key=cv2.contourArea)
        # get coordinates for biggest con
        x_s, y_s, w_s, h_s = cv2.boundingRect(max_con)
        # measure distance for biggest con
        dist_1 = (((x_s+y_s) - (left+top))**2)**0.5

        # get coordinates for 2. biggest con
        x_s, y_s, w_s, h_s = cv2.boundingRect(contours_1[-2])
        # measure distance for 2. biggest con
        dist_2 = (((x_s+y_s) - (left+top))**2)**0.5


        if dist_1 > 100:
            contours_1 = sorted(contours_1, key=cv2.contourArea)
            cv2.fillPoly(img_3c_full, pts=[contours_1[-2]], color=(255, 0, 0))
        else:
            cv2.fillPoly(img_3c_full, pts=[contours_1[-1]], color=(255, 0, 0))




        print(f"roi splash dist1: {dist_1}")



    image = img_3c_full
    return image


def segment_diver(diver_img, roi_box, with_splash):
    image = diver_img
    black_image = np.zeros((np.array(diver_img).shape[0], np.array(diver_img).shape[1], 3)).astype(np.uint8)

    x = roi_box

    left, top = int(x[0]), int(x[1])
    left_w, top_h = int(x[2]), int(x[3])

    # cut roi from full image
    image_roi = image[top:top_h, left:left_w]

    blue_channel = image[:,:,0]
    blue_channel_sum = blue_channel.sum()
    max_blue = 480*640*255
    percentage_blue= round(((blue_channel_sum/max_blue)*100))
    blue_background = False
    contours_inv = []

    if percentage_blue > 70:
        blue_background = True
    print(f"Anteil Blau: {percentage_blue}%\nBlue Background:{str(blue_background)}\nWith Splash:{str(with_splash)}")



    # set treshold
    if with_splash:
        print("Threshold with splash")
        ORANGE_MIN = np.array([0, 0, 0], np.uint8)
        ORANGE_MAX = np.array([255, 63, 197], np.uint8)
        kernel = np.ones((4, 4), np.uint8)
        img_blr = cv2.medianBlur(image_roi, 3)

        hsv_img = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)
        diver_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)

        closing_1 = cv2.morphologyEx(diver_threshed, cv2.MORPH_OPEN, kernel)
        dilation_1 = cv2.dilate(closing_1, kernel, iterations=1)





    elif blue_background and not with_splash:
        print("Threshold with blue background")

        ORANGE_MIN = np.array([24, 0, 0], np.uint8)
        #ORANGE_MAX = np.array([210, 170, 250], np.uint8)
        ORANGE_MAX = np.array([210, 130, 250], np.uint8)


        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_blr = cv2.medianBlur(image_roi, 1)
        hsv_img = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)
        diver_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
        # close to fill holes
        kernel = np.ones((1, 1), np.uint8)
        closing_hsv = cv2.morphologyEx(diver_threshed, cv2.MORPH_CLOSE, kernel)
        #erosion = cv2.erode(closing_1, kernel, iterations=1)



        ###### Grayscale
        # convert to gray
        img_gray_diver = cv2.cvtColor(img_blr, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("2: Grayscale", full_image_gray)
        # stretch contrast histeq
        img_equ_diver = cv2.equalizeHist(img_gray_diver)
        # cv2.imshow("3: Equalize",img_equ)

        # threshold image
        thresh, img_diver_gray_th = cv2.threshold(img_equ_diver, 245, 255, cv2.THRESH_BINARY)

        combined_diver = cv2.bitwise_or(img_diver_gray_th, closing_hsv)
        # show combined diver
        black_image_combined = np.zeros((np.array(diver_img).shape[0], np.array(diver_img).shape[1])).astype(np.uint8)

        black_image_combined[top:top_h, left:left_w] = combined_diver
        cv2.imshow("combined diver", black_image_combined)

        dilation_1 = combined_diver

        """
        # flodd fill
        im_flood_fill = dilation_1.copy()
        h, w = dilation_1.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
        im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
        # find contours on inverted image
        contours_inv, hier_inv = cv2.findContours(im_flood_fill_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_inv = sorted(contours_inv, key=cv2.contourArea)
        """




    else:
        print("Threshold with normal")

        #ORANGE_MIN = np.array([0, 53, 94], np.uint8)
        #ORANGE_MAX = np.array([26, 255, 255], np.uint8)

        ORANGE_MIN = np.array([0, 50, 125], np.uint8)
        ORANGE_MAX = np.array([14, 255, 255], np.uint8)

        kernel_close_hsv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close_gray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close_gray_combined = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        kernel_dilate_hsv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


        kernel_erode = np.ones((1, 1), np.uint8)

        img_blr_hsv = cv2.medianBlur(image_roi, 1)
        img_blr_gray = cv2.medianBlur(image_roi, 3)


        hsv_img = cv2.cvtColor(img_blr_hsv, cv2.COLOR_BGR2HSV)
        diver_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
        diver_threshed = cv2.dilate(diver_threshed, kernel_dilate_hsv, iterations=1)
        diver_threshed_hsv = cv2.morphologyEx(diver_threshed, cv2.MORPH_OPEN, kernel_dilate_hsv)
        #closing_1 = cv2.morphologyEx(diver_threshed, cv2.MORPH_CLOSE, kernel_close_hsv)
        #erosion = cv2.erode(closing_1, kernel_erode, iterations=1)
        #dilation_1 = cv2.dilate(closing_1, kernel, iterations=1)
        ##K-MEANS
        Z = image_roi.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + 1, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        k_means = res.reshape((image_roi.shape))

        img_blr_gray = cv2.medianBlur(k_means, 1)



        ###### Grayscale
        # convert to gray
        img_gray_diver = cv2.cvtColor(img_blr_gray, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("2: Grayscale", full_image_gray)
        # stretch contrast histeq
        img_equ_diver = cv2.equalizeHist(img_gray_diver)
        #cv2.imshow("3: Equalize",img_equ_diver)

        # threshold image
        thresh, img_diver_gray_th = cv2.threshold(img_equ_diver, 254, 255, cv2.THRESH_BINARY)
        img_diver_gray_th = cv2.morphologyEx(img_diver_gray_th, cv2.MORPH_CLOSE, kernel_close_gray)

        #img_diver_gray_th = cv2.bitwise_not(img_diver_gray_th)

        contours_gray_kmeans, hier_inv = cv2.findContours(img_diver_gray_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_gray_kmeans = sorted(contours_gray_kmeans, key=cv2.contourArea)
        for c in contours_gray_kmeans:
            cv2.fillPoly(img_diver_gray_th, pts=[c], color=(0, 0, 0))
        for c in contours_gray_kmeans[-1]:
            cv2.fillPoly(img_diver_gray_th, pts=[c], color=(255, 255, 255))


        combined_diver = cv2.bitwise_or(img_diver_gray_th, diver_threshed_hsv)
        combined_diver = cv2.morphologyEx(combined_diver, cv2.MORPH_CLOSE, kernel_close_gray_combined)
        #combined_diver = cv2.erode(combined_diver, kernel_erode, iterations=1)
        # show combined diver
        black_image_combined = np.zeros((np.array(diver_img).shape[0], np.array(diver_img).shape[1])).astype(np.uint8)
        black_image_thresh_hsv = np.zeros((np.array(diver_img).shape[0], np.array(diver_img).shape[1])).astype(np.uint8)
        black_image_thresh_gray = np.zeros((np.array(diver_img).shape[0], np.array(diver_img).shape[1])).astype(np.uint8)



        black_image_combined[top:top_h, left:left_w] = combined_diver
        black_image_thresh_hsv[top:top_h, left:left_w] = diver_threshed_hsv
        black_image_thresh_gray[top:top_h, left:left_w] = img_diver_gray_th

        #cv2.imshow("diver_threshed_hsv", black_image_thresh_hsv)
        #cv2.imshow("black_image_combined", black_image_combined)
        cv2.imshow("black_image_gray", black_image_thresh_gray)





        dilation_1 = combined_diver
        # flood fill
        im_flood_fill = dilation_1.copy()
        h, w = dilation_1.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
        im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
        # find contours on inverted image
        contours_inv, hier_inv = cv2.findContours(im_flood_fill_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_inv = sorted(contours_inv, key=cv2.contourArea)
        # remove largest element

    # remove large contours
    for i, c in enumerate(contours_inv):
        print(i)
        size = cv2.contourArea(c)
        if size > 500:
            contours_inv.pop(i)




    contours_2, hier_2 = cv2.findContours(dilation_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # generate image
    img_3c_roi = np.zeros((np.array(image_roi).shape[0], np.array(image_roi).shape[1], 3)).astype(np.uint8)
    img_3c_roi[:, :, 0] = dilation_1
    img_3c_roi[:, :, 1] = dilation_1
    img_3c_roi[:, :, 2] = dilation_1

    contours_2 = sorted(contours_2, key=cv2.contourArea)

    for c in contours_2:
        cv2.fillPoly(img_3c_roi, pts=[c], color=(0, 255, 0))
    # fill flood fill inv contoures
    for c in contours_inv:
        cv2.fillPoly(img_3c_roi, pts=[c], color=(0, 255, 0))


    black_image[top:top_h, left:left_w] = img_3c_roi
    return black_image


@torch.no_grad()
def run(weights='/Users/alexander/PycharmProjects/task-2/task2b/best.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    roi_boxes = []
    roi_classes = []


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    img_raw = None
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        if len(pred[0]) < 1:
            im0 = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3)).astype(np.uint8)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_raw = im0.copy()
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    roi_boxes.append(xyxy)
                    roi_classes.append(int(cls))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if True:
                cv2.imshow(str(p), im0)
                if img_raw is None:
                    img_raw = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3)).astype(np.uint8)
                seg = segment(img_raw,roi_boxes,roi_classes)
                cv2.putText(im0,
                            f"Frame: {frame}",(50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 2)

                cv2.imshow("treshed", segment(img_raw,roi_boxes,roi_classes))
                cv2.waitKey(100)  # 1 millisecond
                roi_boxes = []
                roi_classes = []


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/Users/alexander/PycharmProjects/task-2b/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/Users/alexander/PycharmProjects/task-2b/_tigfCJFLZg_00258.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
