namespace OnnxRuntimeYOLOv5Library
{
    public class Parameter
    {
        public static readonly Parameter Default = new Parameter();

        /// <summary>
        /// 이미지 비율 유지 여부 (true 비율 유지, false 1:1)
        /// </summary>
        private bool _keepOriginalAspectRatio;
        public bool KeepOriginalAspectRatio
        {
            get { return _keepOriginalAspectRatio; }
            set { _keepOriginalAspectRatio = value; }
        }
        /// <summary>
        /// 병렬 Inference 여부 (true 싱글, false 병렬)
        /// </summary>
        private bool _suppressParallelInference;
        public bool SuppressParallelInference
        {
            get { return _suppressParallelInference; }
            set { _suppressParallelInference = value; }
        }
        /// <summary>
        /// IoU 값
        /// </summary>
        private float _ioU;
        public float IoU
        {
            get { return _ioU; }
            set { _ioU = value; }
        }
        /// <summary>
        /// 검출 Confidence 값 (=threshold)
        /// </summary>
        private float _confidence;
        public float Confidence
        {
            get { return _confidence; }
            set { _confidence = value; }
        }

        public Parameter()
        {
            _keepOriginalAspectRatio = true;
            _suppressParallelInference = false;
            _ioU = 0.3f;
            _confidence = 0.5f;
        }
    }
}
