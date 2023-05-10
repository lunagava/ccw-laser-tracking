# Import the deep learning library, PyTorch
import torch
import torchvision.transforms as T
#import torchgeometry as tgm
import kornia.geometry as tgm
# Import the spiking neural network library, Norse
import norse

# Import the DVS camera streaming library, AEstream
from aestream import USBInput

# Initialize our canvas
from sdl import create_sdl_surface, events_to_bw

import cv2

import numpy as np

import laser
import time

imagePointsList = []
# 5 points in the grid (4095, 4095) 200 pixels apart from each side and one in the middle
laserPointsList = [(200, 200),
                   # (200, 3895),
                   (3895, 200),
                   (3095, 3095),
                   # (2047, 2047)
                   ]
frame_mat = np.zeros([640, 480], np.uint8)


def onMouse(event, x, y, flags, param):
    global posList
    if event == cv2.EVENT_LBUTTONDOWN:
        imagePointsList.append((x, y))


# Simple 2d gaussian filter/blur
kernel_size = (5, 5)
CameraKernel = T.GaussianBlur(kernel_size, sigma=(0.5, 0.5))
kernel_size = (51, 51)
LaserKernel = T.GaussianBlur(kernel_size, sigma=(10, 10))

# Create horizontal and vertical edge detectors
# kernel_size = 9
# gaussian = torch.sigmoid(torch.linspace(-10, 10, kernel_size + 1))
# kernel = (gaussian.diff() - 1).repeat(kernel_size, 1)
# kernels = CameraKernel#torch.stack((kernel, kernel.T))
# convolution = torch.nn.Conv2d(1, 2, kernel_size, padding=12, bias=False, dilation=3)
# convolution.weight = torch.nn.Parameter(kernels.unsqueeze(1))

# Create Norse network
# - One refractory cell to inhibit pixels
# - One convolutional edge-detection layer
net = norse.torch.SequentialState(
    norse.torch.LIFRefracCell(),
    CameraKernel,
    #    convolution,
    #    norse.torch.LIFCell()
)
state = None  # Start with empty state

net2 = norse.torch.SequentialState(
    LaserKernel
    #    convolution,
    #    norse.torch.LIFCell()
)
state2 = None

OutputLayer = norse.torch.LIFCell()
state3 = None

#laser_size = [4095, 4095]
#devision = 10


coordinates_Laser = (1200, 800)

try:
    # Start streaming from a DVS camera on USB 2:2 and put them on the CPU
    with USBInput((640, 480), device="CPU") as stream:
        with torch.inference_mode():
            # get laser position
            with laser.Laser(DEBUG=False) as l:
                l.on()
                # some function which controls laser and gets affine transformation -> weights for linear projection
                 cv2.namedWindow('eventsImg')
                 for idx, point in enumerate(laserPointsList):
                     l.move(point[0], point[1])
                     time.sleep(0.1)
                     stream.read()
                #     #
                     for i in range(10):
                        l.off()
                        time.sleep(0.03)
                        l.on()
                        time.sleep(0.03)
                    frame_tensor = stream.read()
                    frame_mat = np.array(frame_tensor)
                    cv2.setMouseCallback('eventsImg', onMouse)
                    #
                    while True:
                        cv2.imshow('eventsImg', frame_mat)
                        if cv2.waitKey(1) & 0xFF == 27 or len(imagePointsList) == idx + 1:
                            break
                cv2.destroyAllWindows()
                # cv2.circle(frame_mat, imagePointsList[idx], 5, (0, 0, 255), -1)
                # cv2.destroyAllWindows()

            affine_mat = torch.tensor(cv2.getAffineTransform(np.array([[1,1], [2,3], [3,2]]).astype(np.float32),
                                                             np.array([[1+100,1], [2+100,3], [3+100,2]]).astype(np.float32)
                                                             ).astype(np.float32))
            print(affine_mat)
            window, pixels = create_sdl_surface(640 * 3, 480)

            while True:  # Loop forever
                # Read a tensor (640, 480) tensor from the camera
                tensor = stream.read()

                # get current laser position
                coordinates_Laser = torch.tensor([450, 140])
                # get image of laser
                Laser_image = torch.zeros((1,1, 400, 400))
                Laser_image[0, 0, coordinates_Laser[0]//10, coordinates_Laser[1]//10] = 1
                Laser_image=LaserKernel(Laser_image.view(1, 1, 400, 400))

                # Run the tensor through the network, while updating the state
                with torch.inference_mode():
                    filtered, state = net(tensor.view(1,640, 480), state)
                    TransformedLaser = tgm.warp_affine(filtered.view(1,1,640,480), affine_mat.view(1,2,3), dsize=(400, 400), padding_mode='zeros')
                    filtered2=Laser_image
                    #filtered2, state2 = net2(Laser_image.view(1, 1, 400, 400), state2)
                    filtered3, state3 = OutputLayer(-500 * filtered2 + 5 * TransformedLaser.view(1, 1, 400, 400),
                                                    state3)
                    #print(torch.median(torch.argwhere(filtered3)))
                # Render tensors
                pixels[0:640] = events_to_bw(tensor)  # Input events
                pixels[640: 640 +400, :400] = events_to_bw(filtered2[0, 0])  # First channel
                pixels[640 * 2: 640 * 2+400, :400] = events_to_bw(filtered3[0, 0])  # Second channel
                window.refresh()

finally:
    window.close()





