using Microsoft.ML.OnnxRuntime;
using OnnxRuntimeYOLOv5Library.Enum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxRuntimeYOLOv5Library
{
    public static class Utility
    {
        /// <summary>
        /// 모델 로드 (CPU일 경우 deviceId 미사용)
        /// </summary>
        /// <param name="path">모델 경로</param>
        /// <param name="devices">장치 종류 [CPU, GPU]</param>
        /// <param name="deviceId">GPU 사용 ID</param>
        /// <returns></returns>
        public static InferenceSession LoadModel(string path, Devices devices, int deviceId = 0)
        {
            Debug.Assert(!string.IsNullOrEmpty(path));
            Debug.Assert(File.Exists(path));

            SessionOptions sessionOptions = new SessionOptions();

            switch (devices)
            {
                case Devices.CPU:
                    sessionOptions.AppendExecutionProvider_CPU();
                    break;
                case Devices.GPU:
                    sessionOptions.AppendExecutionProvider_CUDA(deviceId);
                    break;
                default:
                    Debug.Assert(false);
                    break;
            }
            return new InferenceSession(path, sessionOptions);
        }
        /// <summary>
        /// 이미지 로드
        /// </summary>
        /// <param name="path">이미지 경로</param>
        /// <returns></returns>
        public static Image<Rgb24> LoadImage(string path)
        {
            Debug.Assert(!string.IsNullOrEmpty(path));
            Debug.Assert(File.Exists(path));

            return Image.Load<Rgb24>(path);
        }
        /// <summary>
        /// 비트맵에서 Sixlabors (RGB24) 이미지로 변환
        /// </summary>
        /// <param name="bitmap">비트맵 이미지</param>
        /// <returns></returns>
        public static Image<Rgb24> ConvertBitmapToRGB24Image(System.Drawing.Bitmap bitmap)
        {
            Debug.Assert(bitmap != null);

            Image<Rgb24> image = null;
            using (MemoryStream ms = new MemoryStream())
            {
                bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                using (Image img = Image.Load(ms.ToArray()))
                {
                    image = img.CloneAs<Rgb24>();
                }
            }
            return image;
        }
    }
}
