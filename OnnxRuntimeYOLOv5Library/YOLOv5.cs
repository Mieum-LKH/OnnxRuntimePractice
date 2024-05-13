using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxRuntimeYOLOv5Library.Enum;
using OnnxRuntimeYOLOv5Library.Metadata;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace OnnxRuntimeYOLOv5Library
{
    public class YOLOv5
    {

        private bool _initialized;
        public bool initialized
        {
            get { return _initialized; }
        }

        private bool _isRunning;
        public bool IsRunning
        {
            get { return _isRunning; }
        }

        private InferenceSession _inferenceSession;
        public InferenceSession InferenceSession
        {
            get { return _inferenceSession; }
        }

        private CustomMetadata _customMetadata;
        public CustomMetadata CustomMetadata
        {
            get { return _customMetadata; }
        }

        private Parameter _parameter;
        public Parameter Parameter
        {
            get { return _parameter; }
        }

        public YOLOv5()
        {
            _initialized = false;
            _isRunning = false;
            _inferenceSession = null;
            _customMetadata = null;
        }

        public void InitializeOnCPU(string modelPath, Parameter parameter = null)
        {
            Debug.Assert(!string.IsNullOrEmpty(modelPath));
            Debug.Assert(File.Exists(modelPath));
            _inferenceSession = Utility.LoadModel(modelPath, Devices.CPU);
            _customMetadata = CustomMetadata.Parse(_inferenceSession);
            _parameter = parameter != null ? parameter : Parameter.Default;
            _initialized = true;
        }

        public void InitializeOnGPU(string modelPath, int deviceId = 0, Parameter parameter = null)
        {
            Debug.Assert(!string.IsNullOrEmpty(modelPath));
            Debug.Assert(File.Exists(modelPath));
            _inferenceSession = Utility.LoadModel(modelPath, Devices.GPU, deviceId);
            _customMetadata = CustomMetadata.Parse(_inferenceSession);
            _parameter = parameter != null ? parameter : Parameter.Default;
            _initialized = true;
        }
        public void Initialize(InferenceSession inferenceSession, Parameter parameter = null)
        {
            Debug.Assert(inferenceSession != null);
            _inferenceSession = inferenceSession;
            _customMetadata = CustomMetadata.Parse(_inferenceSession);
            _parameter = parameter != null ? parameter : Parameter.Default;
            _initialized = true;
        }


        public IReadOnlyList<Detection> Run(Image<Rgb24> image)
        {
            Debug.Assert(_inferenceSession != null);
            Debug.Assert(image != null);
            if (!_initialized) { return null; }
            if (image == null) { return null; }

            _isRunning = true;

            Size imageOriginalSize = new Size()
            {
                Width = image.Width,
                Height = image.Height
            };

            image.Mutate(x => x.AutoOrient());

            Tensor<float> tensor = Preprocess(image);
            NamedOnnxValue[] onnxValues = MapNamedOnnxValues(new ReadOnlySpan<Tensor<float>>(new Tensor<float>[] { tensor }), new List<string>() { _customMetadata.InputMetadata.Name });
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = Inference(onnxValues, _inferenceSession);
            IReadOnlyList<Detection> detections = Postprocess(new List<NamedOnnxValue>(outputs), imageOriginalSize);
            IReadOnlyList<Detection> supressed = Suppress(detections, _parameter.IoU);

            _isRunning = false;

            return supressed;
        }

        private Tensor<float> Preprocess(Image<Rgb24> image)
        {
            InputMetadata inputMetadata = _customMetadata.InputMetadata;

            Size inputSize = inputMetadata.Size;
            int[] dimension = new int[] { inputMetadata.Batch, inputMetadata.Channel, inputSize.Height, inputSize.Width };
            DenseTensor<float> denseTensor = new DenseTensor<float>(dimension);
            ResizeOptions resizeOptions = new ResizeOptions()
            {
                Size = inputSize,
                Mode = _parameter.KeepOriginalAspectRatio ? ResizeMode.Max : ResizeMode.Stretch
            };

            image.Mutate(x => x.Resize(resizeOptions));

            int xPadding = (inputSize.Width - image.Width) / 2;
            int yPadding = (inputSize.Height - image.Height) / 2;

            int width = image.Width;
            int height = image.Height;

            if (image.DangerousTryGetSinglePixelMemory(out var dangerousMemory))
            {
                Parallel.For(0, width * height, index =>
                {
                    int x = index % width;
                    int y = index / width;

                    Rgb24 pixel = dangerousMemory.Span[index];

                    WritePixel(0, y + yPadding, x + xPadding, pixel, denseTensor);
                });
            }
            else
            {
                Parallel.For(0, height, y =>
                {
                    Span<Rgb24> row = image.DangerousGetPixelRowMemory(y).Span;

                    for (int x = 0; x < width; ++x)
                    {
                        Rgb24 pixel = row[x];

                        WritePixel(0, y + yPadding, x + xPadding, pixel, denseTensor);
                    }
                });
            }
            return denseTensor;
        }

        private void WritePixel(int batch, int y, int x, Rgb24 pixel, DenseTensor<float> target)
        {
            int offsetR = target.Strides[0] * batch
                      + target.Strides[1] * 0
                      + target.Strides[2] * y
                      + target.Strides[3] * x;

            int offsetG = target.Strides[0] * batch
                        + target.Strides[1] * 1
                        + target.Strides[2] * y
                        + target.Strides[3] * x;

            int offsetB = target.Strides[0] * batch
                        + target.Strides[1] * 2
                        + target.Strides[2] * y
                        + target.Strides[3] * x;

            target.Buffer.Span[offsetR] = pixel.R / 255f;
            target.Buffer.Span[offsetG] = pixel.G / 255f;
            target.Buffer.Span[offsetB] = pixel.B / 255f;
        }

        private NamedOnnxValue[] MapNamedOnnxValues(ReadOnlySpan<Tensor<float>> inputs, List<string> names)
        {
            int length = inputs.Length;

            NamedOnnxValue[] values = new NamedOnnxValue[length];

            for (int i = 0; i < length; ++i)
            {
                string name = names[i];

                NamedOnnxValue value = NamedOnnxValue.CreateFromTensor(name, inputs[i]);

                values[i] = value;
            }
            return values;
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(NamedOnnxValue[] inputs, InferenceSession inferenceSession)
        {
            object sync = new object();

            if (_parameter.SuppressParallelInference)
            {
                lock (sync)
                {
                    return inferenceSession.Run(inputs);
                }
            }
            else
            {
                return inferenceSession.Run(inputs);
            }
        }

        private IReadOnlyList<Detection> Postprocess(List<NamedOnnxValue> namedOnnxValues, Size imageOriginalSize)
        {
            Size inputSize = _customMetadata.InputMetadata.Size;

            int xPadding, yPadding;

            if (_parameter.KeepOriginalAspectRatio)
            {
                float reductionRatio = Math.Min(inputSize.Width / (float)imageOriginalSize.Width, inputSize.Height / (float)imageOriginalSize.Height);
                xPadding = (int)((inputSize.Width - imageOriginalSize.Width * reductionRatio) / 2);
                yPadding = (int)((inputSize.Height - imageOriginalSize.Height * reductionRatio) / 2);
            }
            else
            {
                xPadding = 0;
                yPadding = 0;
            }

            float xRatio = (float)imageOriginalSize.Width / inputSize.Width;
            float yRatio = (float)imageOriginalSize.Height / inputSize.Height;

            if (_parameter.KeepOriginalAspectRatio)
            {
                float maxRatio = Math.Max(xRatio, yRatio);
                xRatio = maxRatio;
                yRatio = maxRatio;
            }

            Tensor<float> output = namedOnnxValues[0].AsTensor<float>();
            IReadOnlyList<Detection> detections = Postprocess(output, imageOriginalSize, xPadding, yPadding, xRatio, yRatio);
            return detections;
        }

        private IReadOnlyList<Detection> Postprocess(Tensor<float> output, Size imageOriginalSize, float xPadding, float yPadding, float xRatio, float yRatio)
        {
            int classes = _customMetadata.OutputMetadata.ClassCount;
            int detectionCount = _customMetadata.OutputMetadata.OutputCount;
            List<Detection> detections = new List<Detection>(detectionCount);

            // Object confidence : 실제 Object인지 판별할 수 있는 confidence
            // Class confidence : 어떤 클래스에 속하는지 판별할 수 있는 confidence
            Parallel.For(0, output.Dimensions[1], i =>
            {
                int id = 0;
                float maxScore = 0.0f;
                float obectConfidence = output[0, i, 4];

                for (int j = 0; j < classes; ++j)
                {
                    float classConfidence = output[0, i, j + 5];
                    float max = Math.Max(classConfidence * obectConfidence, maxScore);
                    if (max != maxScore)
                    {
                        id = j;
                        maxScore = max;
                    }
                }

                if (maxScore > _parameter.Confidence)
                {
                    var x = output[0, i, 0];
                    var y = output[0, i, 1];
                    var w = output[0, i, 2];
                    var h = output[0, i, 3];

                    var xMin = (int)((x - w / 2 - xPadding) * xRatio);
                    var yMin = (int)((y - h / 2 - yPadding) * yRatio);
                    var xMax = (int)((x + w / 2 - xPadding) * xRatio);
                    var yMax = (int)((y + h / 2 - yPadding) * yRatio);

                    xMin = Clamp(xMin, 0, imageOriginalSize.Width);
                    yMin = Clamp(yMin, 0, imageOriginalSize.Height);
                    xMax = Clamp(xMax, 0, imageOriginalSize.Width);
                    yMax = Clamp(yMax, 0, imageOriginalSize.Height);

                    var bounds = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);

                    detections.Add(new Detection()
                    {
                        ID = id,
                        Confidence = maxScore,
                        Left = bounds.Left,
                        Top = bounds.Top,
                        Width = bounds.Width,
                        Height = bounds.Height
                    });
                }

            });

            return detections;
        }

        public static int Clamp(int value, int min, int max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        private IReadOnlyList<Detection> Suppress(IReadOnlyList<Detection> detections, float ioU)
        {
            var sorted = detections.OrderByDescending(x => x.Confidence).ToArray();
            var count = sorted.Length;

            var activeCount = count;
            var isActiveDetecions = new bool[count];

            isActiveDetecions.Fill(true);

            var selected = new List<Detection>();
            for (int i = 0; i < count; i++)
            {
                if (isActiveDetecions[i])
                {
                    var detectionA = sorted[i];

                    selected.Add(detectionA);

                    for (var j = i + 1; j < count; j++)
                    {
                        if (isActiveDetecions[j])
                        {
                            var detectionB = sorted[j];

                            if (CalculateIoU(
                                new Rectangle((int)detectionA.Left, (int)detectionA.Top, (int)detectionA.Width, (int)detectionA.Height)
                                , new Rectangle((int)detectionB.Left, (int)detectionB.Top, (int)detectionB.Width, (int)detectionB.Height)) > ioU)
                            {
                                isActiveDetecions[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }
                    if (activeCount <= 0)
                        break;
                }
            }
            return selected;
        }

        public float CalculateIoU(Rectangle first, Rectangle second)
        {
            var areaA = Area(first);

            if (areaA <= 0f)
                return 0f;

            var areaB = Area(second);

            if (areaB <= 0f)
                return 0f;

            var intersection = Rectangle.Intersect(first, second);
            var intersectionArea = Area(intersection);

            return (float)intersectionArea / (areaA + areaB - intersectionArea);
        }

        private static int Area(Rectangle rectangle)
        {
            return rectangle.Width * rectangle.Height;
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
            _customMetadata = null;
            _parameter = null;
            _initialized = false;
            GC.Collect();
        }
    }

    public static class ArrayExtensions
    {
        public static void Fill<T>(this T[] originalArray, T with)
        {
            for (int i = 0; i < originalArray.Length; i++)
            {
                originalArray[i] = with;
            }
        }
    }
}
