# ccw-laser-tracking

yarpserver
docker start gen3_host
docker exec -it gen3_host bash
atis-bridge-sdk --s 60 --filter 0.01
cd hpe-core/example/movenet/build
make
./eros-framer
cd hpe-core/example/movenet
connect the laser through USB to your laptop
export PYTHONPATH=$PYTHONPATH:/home/luna/code/hpe-core:/home/luna/code/yarp/build/lib/python3
python3 movenet_online.py
yarp connect /atis3/AE:o /eroser/AE:i
yarp connect /eroser/img:o /movenet/img:i 

first calibrate to get the affine transformation cicking on the image the three laser points
point the laser by human and see how the galvanic laser follows it

