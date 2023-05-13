#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
#include <event-driven/algs.h>
#include <event-driven/vis.h>
#include <mutex>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;


class eroser : public RFModule, public Thread {

private:

    ev::EROS eros;
    ev::window<ev::AE> input_port;
    yarp::os::BufferedPort<ImageOf<PixelMono> > output_port;
    std::mutex m;
    cv::Mat raw_image;

    resolution res;
    Stamp yarpstamp;
    int freq;

public:

    eroser() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/eroser")).asString()).c_str());

        /* initialize yarp network */
        yarp::os::Network yarp;
        if(!yarp.checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if(!input_port.open(getName("/AE:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        if(!output_port.open(getName("/img:o"))) {
            yError() << "Could not open output port";
            return false;
        }

        //read flags and parameters
        res.height = rf.check("height", Value(480)).asInt32();
        res.width = rf.check("width", Value(640)).asInt32();

//        eros.init(res.width, res.height, 7, 0.3);

        raw_image = cv::Mat::zeros(480,640, CV_8U);

        freq = rf.check("f", Value(200)).asInt32();

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 1.0/freq; //period of synchrnous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.stop();
        output_port.close();
    }

    //synchronous thread
    virtual bool updateModule()
    {

//        static cv::Mat cv_image(res.height, res.width, CV_8U);
//        static cv::Mat eros_copy;

//        m.lock();
//        eros.getSurface().copyTo(eros_copy);
//        m.unlock();

//        cv::GaussianBlur(eros_copy, cv_image, cv::Size(5, 5), 0, 0);

//        cv::GaussianBlur(raw_image, cv_image, cv::Size(5, 5), 0, 0);

//        cv::imshow("img", raw_image);
//        cv::waitKey(1);

        output_port.setEnvelope(yarpstamp);
        output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(raw_image));
        output_port.write();

        raw_image = 0;

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run() {

        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readAll(true);
//            m.lock();
            for (auto &v : input_port)
                raw_image.at<unsigned char>(v.y, v.x) += 1;

//                eros.update(v.x, v.y);
//            m.unlock();
        }
    }

};

int main(int argc, char * argv[])
{
    /* initialize yarp network */
    yarp::os::Network yarp;
    if(!yarp.checkNetwork(2.0)) {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.configure( argc, argv );

    /* create the module */
    eroser instance;
    return instance.runModule(rf);
}
