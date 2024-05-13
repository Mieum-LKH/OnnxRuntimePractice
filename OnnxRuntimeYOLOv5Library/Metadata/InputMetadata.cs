using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace OnnxRuntimeYOLOv5Library.Metadata
{
    public class InputMetadata
    {
        private string _name;
        /// <summary>
        /// InputMetadata name
        /// </summary>
        public string Name { get { return _name; } }
        private int _batch;
        /// <summary>
        /// Image batch size
        /// </summary>
        public int Batch { get { return _batch; } }
        private int _channel;
        /// <summary>
        /// Image channel
        /// </summary>
        public int Channel { get { return _channel; } }
        private Size _imageSize;
        /// <summary>
        /// Image size
        /// </summary>
        public Size Size { get { return _imageSize; } }
        protected InputMetadata(
            string name
            , int batch
            , int channel
            , Size imageSize)
        {
            _name = name;
            _batch = batch;
            _channel = channel;
            _imageSize = imageSize;
        }
        /// <summary>
        /// InputMetadata 변환
        /// </summary>
        /// <param name="names">이름</param>
        /// <param name="metadata">데이터</param>
        /// <returns></returns>
        public static InputMetadata Parse(
            IReadOnlyList<string> names
            , IReadOnlyDictionary<string, NodeMetadata> metadata)
        {
            Debug.Assert(names != null);
            Debug.Assert(metadata != null);
            Debug.Assert(names.Count > 0);
            Debug.Assert(metadata.Count > 0);

            string name = names[0];
            Debug.Assert(metadata.ContainsKey(name));
            NodeMetadata nodeMetadata = metadata[name];
            Debug.Assert(nodeMetadata.Dimensions != null);
            int[] dim = nodeMetadata.Dimensions;
            Debug.Assert(dim.Length == 4);

            return new InputMetadata(name, dim[0], dim[1], new Size(dim[3], dim[2]));
        }

    }
}
