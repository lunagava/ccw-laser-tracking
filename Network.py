# Import the deep learning library, PyTorch
import torch
import torchvision.transforms as T
from torchvision.io import read_image


# Import the spiking neural network library, Norse
import norse

# Import the DVS camera streaming library, AEstream
from aestream import USBInput

# Initialize our canvas
from sdl import create_sdl_surface, events_to_bw

window, pixels = create_sdl_surface(640 * 3, 480)

#Simple 2d gaussian filter/blur
kernel_size = (5,5)
CameraKernel=T.GaussianBlur(kernel_size, sigma=(0.5, 0.5))
kernel_size = (25,25)
LaserKernel=T.GaussianBlur(kernel_size)


# Create horizontal and vertical edge detectors
kernel_size = 9
gaussian = torch.sigmoid(torch.linspace(-10, 10, kernel_size + 1))
kernel = (gaussian.diff() - 1).repeat(kernel_size, 1)
#kernels = CameraKernel#torch.stack((kernel, kernel.T))
#convolution = torch.nn.Conv2d(1, 2, kernel_size, padding=12, bias=False, dilation=3)
#convolution.weight = torch.nn.Parameter(kernels.unsqueeze(1))

# Create Norse network
# - One refractory cell to inhibit pixels
# - One convolutional edge-detection layer
net = norse.torch.SequentialState(
    norse.torch.LIFRefracCell(),
    CameraKernel
#    convolution,
#    norse.torch.LIFCell()
)
state = None  # Start with empty state


net2 = norse.torch.SequentialState(
    	LaserKernel
#    convolution,
#    norse.torch.LIFCell()
)
state2=None


OutputLayer=norse.torch.LIFCell()
state3=None

coordinates_Laser=(450, 140)

try:
    # Start streaming from a DVS camera on USB 2:2 and put them on the CPU
    with USBInput((640, 480), device="CPU") as stream:
        with torch.inference_mode():
            while True:  # Loop forever
                # Read a tensor (640, 480) tensor from the camera
                tensor = stream.read()
                
                #get current laser position
                coordinates_Laser=(0, 0, 450, 140)
                #get image of laser
                Laser_image=torch.zeros((1, 1, 640,480))
                Laser_image[coordinates_Laser]=1
                
                # Run the tensor through the network, while updating the state
                with torch.inference_mode():
                    filtered, state = net(tensor.view(1, 1, 640, 480), state)
                    filtered2, state2 = net2(Laser_image.view(1, 1, 640, 480), state2)
                    filtered3, state3 =OutputLayer(-100000000*filtered2+ 10*filtered.view(1, 1, 640, 480), state3)
                # Render tensors
                pixels[0:640] = events_to_bw(tensor)  # Input events
                pixels[640 : 640 * 2] = events_to_bw(filtered2[0, 0])  # First channel
                pixels[640 * 2 : 640 * 3] = events_to_bw(filtered3[0, 0])  # Second channel
                window.refresh()

finally:
    window.close()
    
    
    
    
    
