#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

void detectFace(CascadeClassifier face_cascade, Mat& frame);

int main()
{
    CascadeClassifier face_cascade;

    string face_cascade_name = "haarcascade_frontalface_alt2.xml";

    if(!face_cascade.load("model/" + face_cascade_name ))
    {
        cout << "[ERROR] Error loading " + face_cascade_name + "\n";
        return -1;
    }
    else
    {
        cout << "[INFO] Successfully loaded face cascade\n";
    }

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "[ERROR] Error initializing video camera!\n" << endl;
        return -1;
    }
    else
    {
        cout << "[INFO] Starting camera...\n";
        cout << "[INFO] Press \'Esc\' to close the program\n";
    }

    char* windowName = "Face Recognition Application";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    while (1)
    {

        Mat frame;
        bool bSuccess = cap.read(frame);

        if (!bSuccess)
        {
            cout << "Error reading frame from camera feed" << endl;
            break;
        }

        flip(frame, frame, 1);

        detectFace(face_cascade, frame);

        imshow(windowName, frame);

        switch (waitKey(30))
        {
        case 27:
            return 0;
        }
    }
    return 0;
}

void detectFace(CascadeClassifier face_cascade, Mat& frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point p1, p2, pText;
        p1.x = faces[i].x;
        p1.y = faces[i].y;
        p2.x = faces[i].x + faces[i].width;
        p2.y = faces[i].y + faces[i].height;
        pText.x = faces[i].x;
        pText.y = faces[i].y - 10;

        rectangle(frame, p1, p2, Scalar(0, 0, 255), 1);
        putText(frame, "unknown", pText, 3, 0.5, Scalar(0, 0, 255), 2);
    }
}
