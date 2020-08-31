#include "util.h"

int main()
{
    CascadeClassifier face_cascade;
    VideoCapture cap(0);
    Mat frame;

    char* window_name = "Face Recognition Application";
    string face_cascade_name = "haarcascade_frontalface_alt2.xml";
    string fn_csv = string("face_db.csv");
    vector<string> names = {"Mark Zuckerberg", "Sundar Pichai", "Andrew Ng"};

    vector<Mat> images;
    vector<int> labels;
    Ptr<face::FaceRecognizer> model;

    // Read the csv file to get all images and its labels
    if (read_csv(fn_csv, images, labels) == -1)
        return -1;

    // Load Haar Cascade for face detection
    if (loadCascade(face_cascade, face_cascade_name) == -1)
        return -1;

    // Check camera availability
    if (checkCamera(cap) == -1)
        return -1;

    // Create LBPHFaceRecognizer with 80 of threshold
    create_model(model, 80, images, labels);

    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (1)
    {
        bool bSuccess = cap.read(frame);

        if (!bSuccess)
        {
            cout << "Error reading frame from camera feed" << endl;
            break;
        }

        // Flip the frame
        flip(frame, frame, 1);

        // Detect and predict all faces in frame
        detect_and_predict(face_cascade, model, frame, names);

        // Show the frame
        imshow(window_name, frame);

        switch (waitKey(30))
        {
        case 27:
            return 0;
        }
    }
    return 0;
}
