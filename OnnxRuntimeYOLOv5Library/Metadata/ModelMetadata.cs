using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxRuntimeYOLOv5Library.Metadata
{
    public class ModelMetadata
    {

        private string _producerName;
        public string ProducerName { get { return _producerName; } }
        private string _graphName;
        public string GraphName { get { return _graphName; } }
        private string _domain;
        public string Domain { get { return _domain; } }
        private string _description;
        public string Description { get { return _description; } }
        private string _graphDescription;
        public string GraphDescription { get { return _graphDescription; } }
        private long _version;
        /// <summary>
        /// 버전
        /// </summary>
        public long Version { get { return _version; } }
        private int _stride;
        /// <summary>
        /// 이미지 Stride
        /// </summary>
        public int Stride { get { return _stride; } }

        private IReadOnlyDictionary<int, string> _classNames;
        /// <summary>
        /// 모델 Class names
        /// </summary>
        public IReadOnlyDictionary<int, string> ClassNames { get { return _classNames; } }

        protected ModelMetadata(
            string producerName
            , string graphName
            , string domain
            , string description
            , string graphDescription
            , long version
            , int stride
            , Dictionary<int, string> classNames)
        {
            _producerName = producerName;
            _graphName = graphName;
            _domain = domain;
            _description = description;
            _graphDescription = graphDescription;
            _version = version;
            _stride = stride;
            _classNames = classNames;
        }
        /// <summary>
        /// 메타데이터 변환 함수
        /// </summary>
        /// <param name="metadata">모델 메타데이터</param>
        /// <returns></returns>
        public static ModelMetadata Parse(Microsoft.ML.OnnxRuntime.ModelMetadata modelMetadata)
        {

            Debug.Assert(modelMetadata != null);
            Debug.Assert(modelMetadata.CustomMetadataMap != null);
            Debug.Assert(modelMetadata.CustomMetadataMap.ContainsKey("stride"));
            Debug.Assert(modelMetadata.CustomMetadataMap.ContainsKey("names"));

            int stride = int.Parse(modelMetadata.CustomMetadataMap["stride"]);
            Dictionary<int, string> names = ParseClassNames(modelMetadata.CustomMetadataMap["names"]);

            return new ModelMetadata(
            modelMetadata.ProducerName
            , modelMetadata.GraphName
            , modelMetadata.Domain
            , modelMetadata.Description
            , modelMetadata.GraphDescription
            , modelMetadata.Version
            , stride
            , names);
        }
        /// <summary>
        /// 모델 클래스 이름들 문자열 변환 함수
        /// </summary>
        /// <param name="classes">클래스 이름들</param>
        /// <returns></returns>
        private static Dictionary<int, string> ParseClassNames(string classes)
        {
            string substr = classes.Substring(1, classes.Length - 2); // [0, abc] -> 123, 456
            string[] split = substr.Split(',');

            Dictionary<int, string> names = new Dictionary<int, string>();

            foreach (var str in split)
            {
                var strSplit = str.Split(':');
                var idx = int.Parse(strSplit[0]);
                var trim = strSplit[1].Trim();
                var name = trim.Substring(1, trim.Length - 2); // 'name' -> name

                names.Add(idx, name);
            }
            return names;
        }
    }
}
