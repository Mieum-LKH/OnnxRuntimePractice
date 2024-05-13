using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxRuntimeYOLOv5Library.Metadata
{
    public class CustomMetadata
    {
        private InputMetadata _inputMetadata;
        /// <summary>
        /// 모델 InputMetadata
        /// </summary>
        public InputMetadata InputMetadata { get { return _inputMetadata; } }
        private ModelMetadata _modelMetadata;
        /// <summary>
        /// 모델 ModelMetadata;
        /// </summary>
        public ModelMetadata ModelMetadata { get { return _modelMetadata; } }
        private OutputMetadata _outputMetadata;
        /// <summary>
        /// 모델 OutputMetadata
        /// </summary>
        public OutputMetadata OutputMetadata { get { return _outputMetadata; } }

        protected CustomMetadata(
            InputMetadata inputMetadata
            , ModelMetadata modelMetadata
            , OutputMetadata outputMetadata)
        {
            _inputMetadata = inputMetadata;
            _modelMetadata = modelMetadata;
            _outputMetadata = outputMetadata;
        }
        /// <summary>
        /// 모델 Metadata 변환 함수
        /// </summary>
        /// <param name="inferenceSession">불러온 모델</param>
        /// <returns></returns>
        public static CustomMetadata Parse(InferenceSession inferenceSession)
        {
            Debug.Assert(inferenceSession != null);
            InputMetadata input = InputMetadata.Parse(inferenceSession.InputNames, inferenceSession.InputMetadata);
            ModelMetadata model = ModelMetadata.Parse(inferenceSession.ModelMetadata);
            OutputMetadata output = OutputMetadata.Parse(inferenceSession.OutputNames, inferenceSession.OutputMetadata);

            return new CustomMetadata(input, model, output);
        }
    }
}
