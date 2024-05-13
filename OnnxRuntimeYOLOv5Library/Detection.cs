namespace OnnxRuntimeYOLOv5Library
{
    public struct Detection
    {
        public int ID { get; set; }
        public float Confidence { get; set; }
        public int Left { get; set; }
        public int Top { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }
}
