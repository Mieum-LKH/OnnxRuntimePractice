using Microsoft.Extensions.Configuration;
using OnnxRuntimeYOLOv5Library;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxRuntimePractice
{
    internal class Program
    {
        static void Main(string[] args)
        {
            IConfiguration configuration = new ConfigurationBuilder()
                .AddCommandLine(args)
                .Build();

            string weights = configuration["weights"];
            string source = configuration["source"];
            string device = configuration["device"];
            string confThres = configuration["conf-thres"];
            string iouThres = configuration["iou-thres"];

            if (string.IsNullOrEmpty(weights))
            {
                Console.WriteLine("Onnx 파일을 입력해야 합니다.");
                return;
            }

            if (!File.Exists(weights))
            {
                Console.WriteLine("Onnx 파일이 없습니다.");
                return;
            }


            float conf = .0f;
            float iou = .0f;

            if (!float.TryParse(confThres, out conf)) { conf = 0.5f; }
            if (!float.TryParse(iouThres, out iou)) { iou = 0.3f; }

            YOLOv5 yOLOv5 = new YOLOv5();

            try
            {
                if (int.TryParse(device, out int result))
                {
                    yOLOv5.InitializeOnGPU(weights, result, new Parameter() { Confidence = conf, IoU = iou });
                }
                else
                {
                    yOLOv5.InitializeOnCPU(weights, new Parameter() { Confidence = conf, IoU = iou });
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Debug.WriteLine(ex.Message);
                Debug.WriteLine(ex.StackTrace);
                Debug.WriteLine(ex.Source);
                return;
            }

            string[] images = null;

            try
            {
                images = Directory.GetFiles(source).Where(file => new string[] { "jpg", "png", "bmp" }.Any(file.ToLower().EndsWith)).ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Debug.WriteLine(ex.Message);
                Debug.WriteLine(ex.StackTrace);
                Debug.WriteLine(ex.Source);
            }

            try
            {
                if (images != null)
                {
                    foreach (string image in images)
                    {
                        Image<Rgb24> img = Image.Load<Rgb24>(image);
                        var results = yOLOv5.Run(img);

                        using (Mat mat = Cv2.ImRead(image))
                        {
                            foreach (var result in results)
                            {
                                mat.Rectangle(new Rect()
                                {
                                    Top = result.Top,
                                    Left = result.Left,
                                    Width = result.Width,
                                    Height = result.Height
                                }, Scalar.RandomColor(), 2, LineTypes.AntiAlias);
                            }

                            Cv2.ImShow("result", mat);
                            Cv2.WaitKey(-1);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Debug.WriteLine(ex.Message);
                Debug.WriteLine(ex.StackTrace);
                Debug.WriteLine(ex.Source);
            }
        }
    }
}
