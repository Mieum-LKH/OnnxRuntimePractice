using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Linq;

namespace OnnxRuntimeYOLOv5Library.Metadata
{
    public class OutputMetadata
    {
        private string _name;
        /// <summary>
        /// OutputMetadata 이름
        /// </summary>
        public string Name { get { return _name; } }
        private int _batch;
        /// <summary>
        /// Image batch size
        /// </summary>
        public int Batch { get { return _batch; } }
        private int _classCount;
        /// <summary>
        /// Model class count
        /// </summary>
        public int ClassCount { get { return _classCount; } }
        private int _outputCount;
        /// <summary>
        /// OutputMetadata output count
        /// </summary>
        public int OutputCount { get { return _outputCount; } }

        protected OutputMetadata(
            string name
            , int batch
            , int classCount
            , int outputCount)
        {
            _name = name;
            _batch = batch;
            _classCount = classCount;
            _outputCount = outputCount;
        }
        /// <summary>
        /// OutputMetadata 변환
        /// </summary>
        /// <param name="names">이름</param>
        /// <param name="metadata">데이터</param>
        /// <returns></returns>
        public static OutputMetadata Parse(
            IReadOnlyList<string> names
            , IReadOnlyDictionary<string, NodeMetadata> metadata)
        {
            Debug.Assert(names != null);
            Debug.Assert(metadata != null);
            Debug.Assert(names.Count != 0);
            Debug.Assert(metadata.Count != 0);

            string name = names[0];
            Debug.Assert(metadata.ContainsKey(name));
            NodeMetadata nodeMetadata = metadata[name];
            Debug.Assert(nodeMetadata.Dimensions != null);
            int[] dim = nodeMetadata.Dimensions;
            Debug.Assert(dim.Length == 3);

            return new OutputMetadata(name, dim[0], dim[2] - 5, dim[1]);
        }
    }
}
