import cv2
import torch
import norse
import torchvision.transforms as T
import numpy as np


def calibrate(Laser, Eventinput,
              laserPointsList = [(1200, 1200),
                       (2895, 1200),
                       (2095, 2095),
                       ],
              devision = 10):
    net = norse.torch.SequentialState(
            norse.torch.LIFRefracCell(),
            T.GaussianBlur((5, 5), sigma=(0.25, 0.25)),
    )
    state = None  # Start with empty state


    imagePointsList = []
    #callback function which accesses the imagePointsList
    def onMouse(event, x, y, flags, param):
        global posList
        if event == cv2.EVENT_LBUTTONDOWN:
            imagePointsList.append((x, y))
    Laser.on()

    cv2.namedWindow('eventsImg')
    with torch.inference_mode():
        for idx, point in enumerate(laserPointsList):
            Laser.move(point[0], point[1])
            cv2.setMouseCallback('eventsImg', onMouse)
            #
            L_State = True
            while True:
                L_State = not L_State
                if L_State:
                    Laser.on()
                else:
                    Laser.off()
                tensor = Eventinput.get()
                filtered, state = net(tensor.view(1, 640, 480), state)
                frame_mat = filtered[0].numpy()
                cv2.imshow('eventsImg', frame_mat)
                if cv2.waitKey(1) & 0xFF == 27 or len(imagePointsList) == idx + 1:
                    break
        cv2.destroyAllWindows()
        return torch.tensor(cv2.getAffineTransform(np.array(imagePointsList).astype(np.float32),
                                                         np.array(laserPointsList).astype(np.float32) / devision
                                                         ).astype(np.float32))


if __name__ == '__main__':
    from example.movenet.laser import Laser
    import Eventconsumer
    import multiprocessing
    import threading

    lock = multiprocessing.Lock()
    Frame = [None]
    ContinueRunning=True


    t1 = threading.Thread(target=Eventconsumer.producer, args=(lock, Frame,ContinueRunning,))
    t1.start()
    Frame=Eventconsumer.Framegetter(lock, Frame)
    # Start streaming from a DVS camera on USB 2:2 and put them on the CPU
    l = Laser(DEBUG=False)
    Affine = calibrate(l, Frame)
    ContinueRunning=False
    print(Affine)
    t1.join()

