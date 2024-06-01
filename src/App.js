import React, { useState, useRef } from "react";
import cv, { log } from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

var frame = null;

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);  // 输入的图像的引用
  const imageRef = useRef(null);    // 图像的引用
  const canvasRef = useRef(null);   // 画布引用

  // Configs
  const modelName = "smoker-detetion.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.2;
  const scoreThreshold = 0.3;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading YOLOv8 Segmentation model", setLoading] // logger
    );
    const yolov8 = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-yolov8.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    );
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms });
    setLoading(null);
  };

  return (
    <div className="App">

      <video id="videoElement" width="640" height="480" autoPlay></video>

      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="btn-container">
        <h1>YOLOv8 Object Detection App</h1>
        <p>
          YOLOv8 object detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const videoElement = document.getElementById('videoElement');
          const canvasElement = document.getElementById('canvas');
          const context = canvasElement.getContext('2d');
          context.drawImage(videoElement, 0, 0, 640, 480);// 将视频画面捕捉后绘制到canvas里面
          const url = canvasElement.toDataURL('image/png');// 将canvas的数据传送到img里

          // url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={async () => {

            const videoElement = document.getElementById('videoElement');

            if (videoElement.srcObject !== '') {
              // 获取用户媒体,包含视频和音频
              navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                  videoElement.srcObject = stream; // 将捕获的视频流传递给video  放弃window.URL.createObjectURL(stream)的使用
                  videoElement.play(); //  播放视频
                });
            }

            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>

      {/* <div className="btn-container">
        <button
          onClick={async () => {
            // const videoElement = document.getElementById('videoElement');
            // //const canvasElement = document.getElementById('canvas');
            // try {
            //   // 获取用户媒体,包含视频和音频
            //   navigator.mediaDevices.getUserMedia({ video: true, audio: true })
            //     .then(stream => {
            //       videoElement.srcObject = stream; // 将捕获的视频流传递给video  放弃window.URL.createObjectURL(stream)的使用
            //       videoElement.play(); //  播放视频
            //     });

            //   //let context = canvasElement.getContext('2d');
            //   //context.drawImage(videoElement, 0, 0, 640, 480);// 将视频画面捕捉后绘制到canvas里面
            //   // imageElement.src = canvasElement.toDataURL('image/png');// 将canvas的数据传送到img里
            //   // imageRef.current.src = canvasElement.toDataURL('image/png');

            // } catch (error) {
            //   console.log('Error accessing camera:', error);
            // }
          }}
        >
          Start camera
        </button>
      </div> */}

    </div>
  );
};

export default App;
