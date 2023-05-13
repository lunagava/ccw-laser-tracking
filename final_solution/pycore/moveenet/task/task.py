
"""
@Fire
https://github.com/fire717
"""
import os
import torch
import numpy as np
import cv2
import json
import time
import csv

import torchvision.transforms as T
import kornia.geometry as tgm
import norse
from example.movenet.laser import Laser
#from pycore.moveenet import laser

# import torch.nn.functional as F

from pycore.moveenet.task.task_tools import movenetDecode, restore_sizes, image_show
from pycore.moveenet.utils.utils import ensure_loc
# from pycore.moveenet.visualization.visualization import superimpose_pose
from pycore.moveenet.utils.metrics import pck
from datasets.h36m.utils.parsing import movenet_to_hpecore


class Task():
    def __init__(self, cfg, model):

        self.cfg = cfg
        self.init_epoch = 0
        if(self.cfg["GPU_ID"]):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(self.device)

        net = norse.torch.SequentialState(
            #norse.torch.LIFRefracCell(),
            T.GaussianBlur(kernel_size = (5, 5), sigma=(0.5, 0.5)),
            #
        )

        self.net=net
        self.state = None
        self.LaserKernel=T.GaussianBlur((101, 101), sigma=(20, 20))
        self.OutputLayer = norse.torch.LIFCell()
        self.state3 = None

        self.affine_mat = None
        # self.affine_mat=torch.tensor([[1.0, 0.0, 0.0],
        #                              [0.0, 1.0, 0.0]]).to(self.device)
        self.image_points = []

        self.UseLaser=True
        self.Laser_pos = (2000, 2000)
        if self.UseLaser:
            self.Laser=Laser(DEBUG=False)
            self.Laser.on()
            self.Laser.move(*self.Laser_pos)
        cv2.namedWindow('eventsImg')

        # callback function which accesses the imagePointsList
        def onMouse(event, x, y, flags, param):
            global posList
            if event == cv2.EVENT_LBUTTONDOWN:
                self.image_points.append((x, y))
        cv2.setMouseCallback('eventsImg', onMouse)



        #self.model = model.to(self.device)

        ############################################################
        # loss
        #self.loss_func = MovenetLoss(self.cfg)

        # optimizer
        #self.optimizer = getOptimizer(self.cfg['optimizer'],
        #                              self.model,
        #                              self.cfg['learning_rate'],
        #                              self.cfg['weight_decay'])

        #self.val_losses = np.zeros([20])
        #self.early_stop = 0
        #self.val_loss_best = np.Inf

        # scheduler
        #self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

        # ensure_loc(os.path.join(self.cfg['save_dir'], self.cfg['label']))

    def predict_online(self, img_in, write_csv=None, ts = None):
        # print("Hey, here I am!")
        #self.model.eval()
        correct = 0
        size = self.cfg["img_size"]
        # print(size)

        if (self.affine_mat == None):
            laserPointsList = [(1200, 1200),
                               (2895, 1200),
                               (2095, 2095)]
            devision = 10

            current_point = laserPointsList[len(self.image_points)]
            if (self.Laser_pos[0] != current_point[0]) or (self.Laser_pos[1] != current_point[1]):
                if self.UseLaser:
                    self.Laser_pos = self.Laser.move(int(current_point[0]), int(current_point[1]))
                else:
                    self.Laser_pos = [int(current_point[0]), int(current_point[1])]

            img_in = 255*(img_in/np.max(img_in))
            cv2.imshow('eventsImg', img_in)
            if cv2.waitKey(1) & 0xFF == 27 or len(self.image_points) == 3:
                cv2.destroyAllWindows()
                self.affine_mat = torch.tensor(cv2.getAffineTransform(np.array(self.image_points).astype(np.float32),
                                                         np.array(laserPointsList).astype(np.float32) / devision
                                                         ).astype(np.float32))
                print(self.affine_mat)
                self.affine_mat = self.affine_mat.to(self.device)
        else:
            with torch.inference_mode(): #torch.no_grad():

                img_size_original = img_in.shape

                #img_in = 255*(img_in/np.max(img_in))

                #cv2.imshow('image input', (255*img_in/np.max(img_in)))
                #input_image_resized = np.zeros([1, 3, size, size])
                # print(input_image_resized.shape)

                #input_image = cv2.resize(img_in, (size, size))
                #input_image_resized[0, 0, :, :] = input_image[:, :]
                #input_image_resized[0, 1, :, :] = input_image[:, :]
                #input_image_resized[0, 2, :, :] = input_image[:, :]
                #input_image_resized = input_image_resized.astype(np.float32)


                #img = torch.from_numpy(input_image_resized)
                #img = torch.from_numpy(np.transpose(img_in.astype(np.float32)))
                img = torch.from_numpy(img_in.astype(np.float32))


                Laser_image = torch.zeros((1, 1, 400, 400))
                Laser_image[0, 0, self.Laser_pos[0] // 10, self.Laser_pos[1] // 10] = 255
                Laser_image = Laser_image.to(self.device)
                Laser_image = self.LaserKernel(Laser_image.view(1, 1, 400, 400))

                # print(np.sum(Laser_image.cpu().numpy()))

                #cv2.imshow('laser', Laser_image.view(400,400).cpu().numpy())

                tensor = img.to(self.device)
                filtered, self.state = self.net(tensor.view(1, 480, 640 ), self.state)
                TransformedLaser = tgm.warp_affine(filtered.view(1, 1, 480, 640), self.affine_mat.view(1, 2, 3),
                                                   dsize=(400, 400), padding_mode='zeros')



                filtered3, self.state3 = self.OutputLayer(-50 * Laser_image.view(1, 1, 400, 400) + 1 * TransformedLaser.view(1, 1, 400, 400),
                                                self.state3)

                #print(np.unique(filtered3.view(400, 400).cpu().numpy()))
                #print(self.state3)

                laserposition = np.round(np.mean(torch.argwhere(filtered3[0, 0]).cpu().numpy(), axis=0))

                #print(laserposition[0])
                #print(laserposition[1])

                final_img = (255 * filtered3.view(400, 400).cpu().numpy())

                #final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
                # Laser_space=filtered3[0, 0].cpu().numpy()
                # total_mass=np.sum(Laser_space)
                # if total_mass>5:
                #     Positions=np.argwhere(Laser_space)
                #     print(Laser_space.shape)
                #     print(Laser_space[Positions])
                #     laserposition=(Positions*Laser_space[Positions])/total_mass

                #laserposition = torch.argwhere(filtered3[0, 0]).cpu().numpy()


                #print(laserposition)
                if laserposition[0] > 0 and laserposition[1] > 0:
                    #cv2.circle(final_img_rgb, (int(laserposition[1]),int(laserposition[0])), 5, (0, 0, 255), -1)
                    if self.UseLaser:
                        #self.Laser_pos = self.Laser.move(int(laserposition[0] * 10), int(laserposition[1] * 10))
                        self.Laser_pos = self.Laser.move(int(laserposition[1] * 10), int(laserposition[0] * 10))

                    else:
                        self.Laser_pos = [10*int(laserposition[0]), 10*int(laserposition[1])]

                output = filtered3

                #cv2.imshow('Network1', final_img_rgb)
                #cv2.waitKey(1)

                # img_size_original = img.shape
                #img = img.to(self.device)
                #start_sample = time.time()
                #output = self.model(img)
                instant = {}


                return instant


    def evaluate(self, data_loader,fastmode=False):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, img_size_original, ts) in enumerate(data_loader):
                if img_size_original == 0:
                    continue

                start_sample = time.time()
                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    if not fastmode:
                        acc_intermediate = correct_kps / total_kps
                        acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                        print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                        # print('Time since beginning:', time.time()-start)
                        print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if not fastmode:
                    if torso_diameter is None:
                        pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    else:
                        pck_acc = pck(pre, gt, torso_diameter, threshold=0.5, num_classes=self.cfg["num_classes"],
                                      mode='torso')
                    correct_kps += pck_acc["total_correct"]
                    total_kps += pck_acc["total_keypoints"]
                    joint_correct += pck_acc["correct_per_joint"]
                    joint_total += pck_acc["anno_keypoints_per_joint"]

                    img_out, pose_gt = restore_sizes(imgs[0], gt, (int(img_size_original[0]), int(img_size_original[1])))
                    # print('gt after restore function', pose_gt)

                _, pose_pre = restore_sizes(imgs[0], pre, (int(img_size_original[0]), int(img_size_original[1])))
                # print('pre after restore function',pose_pre)

                kps_2d = np.reshape(pose_pre, [-1, 2])
                kps_hpecore = movenet_to_hpecore(kps_2d)
                kps_pre_hpecore = np.reshape(kps_hpecore, [-1])
                if self.cfg['write_output']:
                    row = self.create_row(ts,kps_pre_hpecore, delay=time.time()-start_sample)
                    sample = '_'.join(os.path.basename(img_names[0]).split('_')[:-1])
                    write_path = os.path.join(self.cfg['results_path'],self.cfg['dataset'],sample,'movenet.csv')
                    ensure_loc(os.path.dirname(write_path))
                    self.write_results(write_path, row)

        if not fastmode:
            acc = correct_kps / total_kps
            acc_joint_mean = np.mean(joint_correct / joint_total)
            print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc))
            print('[Info] Mean Joint Acc: {:.3f}% \n'.format(100. * acc_joint_mean))

    def infer_video(self, data_loader, video_path):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        size = self.cfg["img_size"]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (size * 2, size * 2))

        text_location = (10, size * 2 - 10)  # bottom left corner of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        fontColor = (0, 0, 255)
        thickness = 1
        lineType = 2

        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, _, _) in enumerate(
                    data_loader):

                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    acc_intermediate = correct_kps / total_kps
                    acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                    # print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                    print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                    print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if self.cfg['dataset'] in ['coco', 'mpii']:
                    pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    th_val = head_size_norm
                else:
                    pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')
                    th_val = torso_diameter

                correct_kps += pck_acc["total_correct"]
                total_kps += pck_acc["total_keypoints"]
                joint_correct += pck_acc["correct_per_joint"]
                joint_total += pck_acc["anno_keypoints_per_joint"]

                img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]

                for i in range(len(gt[0]) // 2):
                    # x = int(gt[0][i * 2] * w)
                    # y = int(gt[0][i * 2 + 1] * h)
                    # cv2.circle(img, (x, y), 2, (0, 255, 0), 1)  # gt keypoints in green

                    x = int(pre[0][i * 2] * w)
                    y = int(pre[0][i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 2, (0, 0, 255), 1)  # predicted keypoints in red

                img2 = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)
                # str = "acc: %.2f, th: %.2f " % (pck_acc["total_correct"] / pck_acc["total_keypoints"], th_val)
                # cv2.putText(img2, str,
                #             text_location,
                #             font,
                #             fontScale,
                #             fontColor,
                #             thickness,
                #             lineType)
                # cv2.line(img2, [10, 10], [10 + int(head_size_norm * 2), 10], [0, 0, 255], 3)
                # cv2.imshow("prediction", img)
                # cv2.waitKey(10)
                # basename = os.path.basename(img_names[0])
                # ensure_loc('eval_result')
                # cv2.imwrite(os.path.join('eval_result', basename), img)
                img2 = np.uint8(img2)
                out.write(img2)

        acc = correct_kps / total_kps
        acc_joint_mean = np.mean(joint_correct / joint_total)
        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc))
        print('[Info] Mean Joint Acc: {:.3f}% \n'.format(100. * acc_joint_mean))
        out.release()

    def modelLoad(self, model_path, data_parallel=False):

        return 0
        if os.path.splitext(model_path)[-1] == '.json':
            with open(model_path, 'r') as f:
                model_path = json.loads(f.readlines()[0])
                str1 = ''
            init_epoch = int(str1.join(os.path.basename(model_path).split('_')[0][1:]))
            self.init_epoch = init_epoch
        print(model_path)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print('The checkpoint expected does not exist. Please check the path and filename.')

        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def write_results(self, path, row):
        # Write a data point into a csvfile
        with open(path, 'a') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(row)

    def create_row(self, ts, skt, delay = 0.0):
        # Function to create a row to be written into a csv file.
        row = []
        ts = float(ts)
        row.extend([ts, delay])
        row.extend(skt)
        return row
