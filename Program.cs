using System;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;

class Program
{
    static void Main()
    {
        string testImagePath = "C:/Users/User/Desktop/mouse_bite/mousebite/bin/Debug/net8.0/test/2.jpg";

        // Read the image
        Image<Bgr, Byte> img = new Image<Bgr, Byte>(testImagePath);

        // Convert to grayscale
        Image<Gray, Byte> gray = img.Convert<Gray, Byte>();
        gray = gray.SmoothGaussian(5);

        // Apply binary thresholding
        Image<Gray, Byte> binary = gray.ThresholdBinary(new Gray(50), new Gray(255));
        CvInvoke.Imwrite("A1_binary.png", binary);

        // Morphological closing to fill small holes
        Mat kernelClose = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(9, 9), new Point(-1, -1));
        Mat closed = new Mat();
        CvInvoke.MorphologyEx(binary, closed, MorphOp.Close, kernelClose, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        CvInvoke.Imwrite("A2_closed.png", closed);

        // Create structuring element larger than mouse bites
        Mat kernelLarge = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(4, 4), new Point(-1, -1));

        var closed_second = closed.ToImage<Gray, Byte>();
        closed_second = closed_second.SmoothGaussian(5);

        // Dilate back to original size
        Mat dilated = new Mat();
        CvInvoke.Dilate(closed, dilated, kernelLarge, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));
        CvInvoke.Imwrite("A4_dilated.png", dilated);



        // Read the dilated
        Image<Gray, Byte> dilated_new = new Image<Gray, Byte>(dilated);

        // Convert to grayscale
        Image<Gray, Byte> gray_new = dilated_new.Convert<Gray, Byte>();

        // Apply Gaussian smoothing to reduce noise
        gray_new = gray_new.SmoothGaussian(3);

        // Apply Canny edge detection
        Image<Gray, Byte> edges_new = gray_new.Canny(30, 150);

        // Save the result (no inversion is needed, as Canny produces white edges on a black background)
        CvInvoke.Imwrite("A4.1_edges_white.png", edges_new);


        kernelClose = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(9, 9), new Point(-1, -1));
        Mat closed_new = new Mat();
        CvInvoke.MorphologyEx(edges_new, closed_new, MorphOp.Close, kernelClose, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        CvInvoke.Imwrite("A4.2_closed.png", closed_new);


        /*var dilated_new = dilated.ToImage<Gray, Byte>();

        Image<Gray, Byte> Inverted = dilated_new.Not();
        CvInvoke.Imwrite("A5_inverted.png", Inverted);




        CvInvoke.Dilate(Inverted, dilated, kernelLarge, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));
        CvInvoke.Imwrite("A5.1_dilated.png", dilated);
        */


        // Find contours of mouse bites
        var contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
        CvInvoke.FindContours(dilated, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

        // Draw contours on original image
        Image<Bgr, Byte> result = img.Clone();
        for (int i = 0; i < contours.Size; i++)
        {
            CvInvoke.DrawContours(result, contours, i, new MCvScalar(0, 0, 255), 10);
        }

        // Save the result with mouse bites marked
        CvInvoke.Imwrite("A6_result_with_mouse_bites.png", result);
        Console.WriteLine("Mouse bites detected and marked on the image.");

        // Now, let's detect parallel lines on the result image



    }
}
