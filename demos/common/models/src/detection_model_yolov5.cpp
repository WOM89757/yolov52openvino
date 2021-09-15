/*
// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/detection_model_yolov5.h"
#include <utils/slog.hpp>
#include <utils/common.hpp>
#include <utils/ocv_common.hpp>
#include <utils/nms.hpp>
#include <ngraph/ngraph.hpp>
// #include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace InferenceEngine;



/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlobDiv(const cv::Mat& orig_image, const InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    if (static_cast<size_t>(orig_image.channels()) != channels) {
        throw std::runtime_error("The number of channels for net input and image must match");
    }
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    T* blob_data = blobMapped.as<T*>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
            static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    if (channels == 1) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
            }
        }
    } else if (channels == 3) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t  h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] =
                            resized_image.at<cv::Vec3b>(h, w)[c] / 255.0f;
                            // resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

ModelYolo5::ModelYolo5(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize,
    bool useAdvancedPostprocessing, float boxIOUThreshold, const std::vector<std::string>& labels) :
    DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels),
    boxIOUThreshold(boxIOUThreshold),
    useAdvancedPostprocessing(useAdvancedPostprocessing) {
}

void ModelYolo5::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    slog::info << "inputInfo.size() " << inputInfo.size() << slog::endl;
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InputInfo::Ptr& input = inputInfo.begin()->second;
    inputsNames.push_back(inputInfo.begin()->first);
     //yolov4
    // input->setPrecision(Precision::U8);
    // if (useAutoResize) {
    //     input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    //     input->getInputData()->setLayout(Layout::NHWC);
    // }
    // else {
    //     input->getInputData()->setLayout(Layout::NCHW);
    // }

    input->setPrecision(Precision::FP32);
    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
        slog::info << "111" << slog::endl;
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
        ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
        cnnNetwork.reshape(inputShapes);
    }

// throw std::logic_error("This demo accepts networks that have only one input");


    //--- Reading image input parameters
    const TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto& output : outputInfo) {
        output.second->setPrecision(Precision::FP32);
        //yolov4
        // output.second->setLayout(Layout::NCHW);
        outputsNames.push_back(output.first);
    }

    if (auto ngraphFunction = (cnnNetwork).getFunction()) {
        int maskCount = 0;
        for (const auto op : ngraphFunction->get_ops()) {
            auto outputLayer = outputInfo.find(op->get_friendly_name());

            if (outputLayer != outputInfo.end()) {

                slog::info << "op->get_friendly_name():\t" << op->get_friendly_name() << "\t"; 
                slog::info << "op->get_shape():\t" << op->get_shape() << "\t" << op->get_shape().at(2) << "\t"; 
                slog::info << "op->get_output_tensor():\t" << op->get_output_tensor(0).get_shape() << "\t"; 
                slog::info << "get_output_size:\t" << op->get_output_size() << slog::endl; 

                // throw std::runtime_error("here");

                RegionYolov5 r;
                r.num = 3;
                r.classes = 80;
                r.coords = 4;
                // r.sides[0] = op->get_shape().at(0);
                // r.sides[1] = op->get_shape().at(0);


                // std::vector<float> anchors = {116,90, 156,198, 373,326,
                //                               30,61, 62,45, 59,119,
                //                               10,13, 16,30, 33,23,
                //                               };

                // yolov5
                std::vector<float> anchors = {10,13, 16,30, 33,23,
                                              30,61, 62,45, 59,119,
                                              116,90, 156,198, 373,326};
                std::vector<int> netGrids = {80, 40, 20};

                // std::vector<float> anchors = {116,90, 156,198, 373,326,
                //                               10,13, 16,30, 33,23,
                //                               30,61, 62,45, 59,119,
                //                               };
                // std::vector<int> netGrids = {20, 40, 80};

                // anchors:
                // - [10,13, 16,30, 33,23]  # P3/8
                // - [30,61, 62,45, 59,119]  # P4/16
                // - [116,90, 156,198, 373,326]  # P5/32

                std::vector<std::vector<int> > masks = {{0, 1, 2}, {3, 4, 5},{6, 7, 8},};
                r.netGrid = netGrids[maskCount];
                
                std::vector<int> masked_anchors;
                for (auto idx : masks[maskCount]) {
                    masked_anchors.push_back(anchors[idx * 2]);
                    masked_anchors.push_back(anchors[idx * 2 + 1]);
                }
                r.anchors = masked_anchors;
                maskCount++;
                regions.emplace(outputLayer->first, RegionYolov5(r));
            }
        }
        slog::info << "finished add regions " << slog::endl;
    }
    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }
}

std::shared_ptr<InternalModelData> ModelYolo5::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {

    // auto inframe = inputData.asRef<ImageInputData>().inputImage;
    // // cv::imwrite("00.jpg", inframeData.inputImage);
    // // cv::Mat &inframe = inframeData.inputImage;
    // // cv::imwrite("0.jpg", inframe);
    // cv::resize(inframe, inframe, cv::Size(640,640));
    // // cv::imwrite("1.jpg", inframe);
    // cv::cvtColor(inframe, inframe, cv::COLOR_BGR2RGB);
    // // cv::imwrite("2.jpg", inframe);
    
    // size_t img_size = 640 * 640;
    // if (inputsNames.size() >= 1) {
    //     auto blob = request->GetBlob(inputsNames[0]);
    //     LockedMemory<void> blobMapped = as<MemoryBlob>(blob)->wmap();
    //     auto data = blobMapped.as<float*>();
    //     //nchw
    //     for(size_t row = 0; row < 640; row++){
    //         for(size_t col = 0; col < 640; col++){
    //             for(size_t ch = 0; ch < 3; ch++){
    //                 // data[img_size * ch + row * 640 + col] = float(inframe.at<cv::Vec3b>(row, col)[ch]);
    //                 data[img_size * ch + row * 640 + col] = float(inframe.at<cv::Vec3b>(row, col)[ch]) / 255.0f;
    //             }
    //         }
    //     }
    // }
    
    // cv::imwrite("01.jpg", inframeData.inputImage);
    // cv::imwrite("02.jpg", inframe);

    // throw std::runtime_error("s");
    // return DetectionModel::preprocess(inputData, request);


    auto& img = inputData.asRef<ImageInputData>().inputImage;


	cv::Mat resize_img = letterBox(img);

    // if (useAutoResize) {
    //     /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
    //     request->SetBlob(inputsNames[0], wrapMat2Blob(img));
    //     /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
    //     return std::make_shared<InternalImageMatModelData>(img);
    // }
    /* Resize and copy data from the image to the input blob */
    InferenceEngine::Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
    // matU8ToBlob<uint8_t>(img, frameBlob);

    matU8ToBlobDiv<float>(resize_img, frameBlob);

    // size_t img_size = 640 * 640;
    // if (inputsNames.size() >= 1) {
    //     auto blob = request->GetBlob(inputsNames[0]);
    //     LockedMemory<void> blobMapped = as<MemoryBlob>(blob)->wmap();
    //     auto data = blobMapped.as<float*>();
    //     //nchw
    //     for(size_t row = 0; row < 640; row++){
    //         for(size_t col = 0; col < 640; col++){
    //             for(size_t ch = 0; ch < 3; ch++){
    //                 // data[img_size * ch + row * 640 + col] = float(inframe.at<cv::Vec3b>(row, col)[ch]);
    //                 data[img_size * ch + row * 640 + col] = float(inframe.at<cv::Vec3b>(row, col)[ch]) / 255.0f;
    //             }
    //         }
    //     }
    // }



        // cv::imwrite("image1.jpg", img);
    // InferenceEngine::SizeVector blobSize = frameBlob->getTensorDesc().getDims();
    // const size_t width = blobSize[3];
    // std::cout << "blob width : " << width << std::endl;
    // std::cout << "img.size().h: " << img.size().height << std::endl;
    // std::cout << "img.size().width: " << img.size().width << std::endl;
    

    // throw std::runtime_error("s");
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}


std::unique_ptr<ResultBase> ModelYolo5::postprocess(InferenceResult & infResult) {
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);

    // auto inframe = result->metaData->asRef<ImageMetaData>().img;
    // cv::imwrite("f1.jpg", inframe);

    // throw std::runtime_error("ss");
    std::vector<DetectedObject> objects;

    std::vector<cv::Rect> origin_rect;
	std::vector<float> origin_rect_cof;
	std::vector<int> classId;

    // Parsing outputs
    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    for (auto& output : infResult.outputsData) {
        this->parseYOLOV5Output(output.first, output.second, netInputHeight, netInputWidth,
            internalData.inputImgHeight, internalData.inputImgWidth, origin_rect, origin_rect_cof, classId);
        // slog::info << "layer name: " << output.first << " origin_rect size " << origin_rect.size() << slog::endl;
    }

    // slog::info << "origin_rect size " << origin_rect.size() << slog::endl;
    std::vector<int> final_id;
    // dnn::NMSBoxes(origin_rect,origin_rect_cof,_cof_threshold,_nms_area_threshold,final_id);
    // slog::info << "confidenceThreshold " << confidenceThreshold << slog::endl;
    // for(size_t i = 0; i < origin_rect.size(); ++i) {
    //     std::cout << "o_rect-" << i << "\t" << origin_rect[i] << "\tprod: " << origin_rect_cof[i] << std::endl;
    // }
    cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, confidenceThreshold, 0.5, final_id);
    // slog::info << "final_id size " << final_id.size() << slog::endl;
    // throw std::runtime_error("s");


    //根据final_id获取最终结果
    for(size_t i = 0; i < final_id.size(); ++i){
        cv::Rect resize_rect= origin_rect[final_id[i]];
        //调用detect2origin方法将框映射到原图
		cv::Rect rawrect = detect2origin(resize_rect, ratio, topPad, leftPad);
		// cv::Rect rawrect = resize_rect;
		//结果以数据结构Object保存到vector
		// std::cout << "is: " << final_id[i] << std::endl;
		// std::cout << "is: " << classId[final_id[i]] << std::endl;

        DetectedObject obj;
        obj.x = rawrect.x;
        obj.y = rawrect.y;
        obj.width = rawrect.width;
        obj.height = rawrect.height;

        obj.confidence = origin_rect_cof[final_id[i]];
        obj.labelID = classId[final_id[i]];
        obj.label = getLabelName(obj.labelID - 5);
        objects.push_back(obj);
    }
    // slog::info << "objects size " << objects.size() << slog::endl;
    // slog::info << slog::endl << slog::endl;
    if (useAdvancedPostprocessing) {
        // Advanced postprocessing
        // Checking IOU threshold conformance
        // For every i-th object we're finding all objects it intersects with, and comparing confidence
        // If i-th object has greater confidence than all others, we include it into result
        for (const auto& obj1 : objects) {
            bool isGoodResult = true;
            for (const auto& obj2 : objects) {

                if (obj1.labelID == obj2.labelID && obj1.confidence < obj2.confidence) { // if obj1 is the same as obj2, condition expression will evaluate to false anyway
                // if (obj1.labelID == obj2.labelID && obj1.confidence < obj2.confidence && intersectionOverUnion(obj1, obj2) >= boxIOUThreshold) { // if obj1 is the same as obj2, condition expression will evaluate to false anyway
                    // slog::info << "intersectionOverUnion " << intersectionOverUnion(obj1, obj2) << " boxIOUThreshold " << boxIOUThreshold << slog::endl;

                    if (intersectionOverUnion(obj1, obj2) < 0.5) continue;
                    
                    isGoodResult = false;
                    break;
                }
            }
            if (isGoodResult) {
                result->objects.push_back(obj1);
            }
        }
        // slog::info << "result->objects size " << result->objects.size() << slog::endl;
    } else {
        // Classic postprocessing
        std::sort(objects.begin(), objects.end(), [](const DetectedObject& x, const DetectedObject& y) { return x.confidence > y.confidence; });
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].confidence == 0)
                continue;
            for (size_t j = i + 1; j < objects.size(); ++j)
                if (intersectionOverUnion(objects[i], objects[j]) >= boxIOUThreshold)
                    objects[j].confidence = 0;
            result->objects.push_back(objects[i]);
        }
    }

    return std::unique_ptr<ResultBase>(result);
}

void ModelYolo5::parseYOLOV5Output(const std::string& output_name,
    const InferenceEngine::Blob::Ptr& blob, const unsigned long resized_im_h,
    const unsigned long resized_im_w, const unsigned long original_im_h,
    const unsigned long original_im_w,
    std::vector<cv::Rect> &origin_rect,
	std::vector<float> &origin_rect_cof,
    std::vector<int>& classId) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    // std::cout << "out_blob_h : " << out_blob_h << " w " << out_blob_w << std::endl;
    if (out_blob_h != out_blob_w) {
        throw std::runtime_error("Invalid size of output " + output_name +
            " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
            ", current W = " + std::to_string(out_blob_h));
    }

    // --------------------------- Extracting layer parameters -------------------------------------
    auto it = regions.find(output_name);
    if(it == regions.end()) {
        throw std::runtime_error(std::string("Can't find output layer with name ") + output_name);
    }
    auto& region = it->second;
    // slog::info << " layer name: " << it->first << slog::endl;

    // auto side = out_blob_h;
    // auto side_square = side * side;

    // std::cout << "output_blob size：" << blob->size() << std::endl;
    // std::cout << "output_blob element_size：" << blob->element_size() << std::endl;
    // std::cout << "region.netGrid ：" << region.netGrid << std::endl;

    // --------------------------- Parsing YOLOV5 Region output -------------------------------------

    std::vector<int> anchors = region.anchors;
    // std::cout << "anchors ：" << anchors[0] << " " << anchors[1] << " " << anchors[2] << " " << anchors[3] << " " << anchors[4] << " " << anchors[5] << std::endl;

    // const float* output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();

    int item_size = 85;
    size_t anchor_n = 3;
    int net_grid = region.netGrid;
                
    // std::cout << "net_grid " << net_grid << std::endl;
    int count = 0;
    // if (net_grid != 40) return;
    for(int i = 0; i < net_grid; ++i) {
        for(int j = 0; j < net_grid; ++j) {
            for(size_t n = 0; n < anchor_n; ++n) {

                double box_prob = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 4];
                // std::cout << "box_prob : " << box_prob << std::endl;

                box_prob = sigmoid(box_prob);
                //框置信度不满足则整体置信度不满足
                if(box_prob < confidenceThreshold)
                    continue;
                // std::cout << "box_prob: " << box_prob << " confidenceThreshold: " << confidenceThreshold << std::endl;

                
                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 0];
                double y = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 1];
                double w = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 2];
                double h = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 3];
                
                double max_prob = 0;
                int labelMaxID = 0;
                for(int t = 5; t < 85; ++t){
                    double tp= output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + t];
                    tp = sigmoid(tp);
                    if(tp > max_prob){
                        max_prob = tp;
                        labelMaxID = t;
                    }
                }
                // std::cout << "max_prob: " << max_prob << std::endl;


                float cof = box_prob * max_prob;                
                // float cof = box_prob;                
                // //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                if(cof < confidenceThreshold)
                    continue;
                // std::cout << "box_prob: " << box_prob << " confidenceThreshold: " << confidenceThreshold << std::endl;
                // std::cout << "cof: " << cof << " labelMaxID: " << labelMaxID << std::endl;


                x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / net_grid;
                y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / net_grid;
                w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
                h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

                double r_x = x - w / 2;
                double r_y = y - h / 2;
                cv::Rect rect = cv::Rect(round(r_x),round(r_y),round(w),round(h));
                origin_rect.push_back(rect);
                origin_rect_cof.push_back(cof);
                classId.push_back(labelMaxID);
                count++;
            }
        } 
    }

    // slog::info << "o_rect size " << origin_rect.size() << slog::endl;
    // slog::info << "o_rect_cof size " << origin_rect_cof.size() << slog::endl;
}


int ModelYolo5::calculateEntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

double ModelYolo5::intersectionOverUnion(const DetectedObject& o1, const DetectedObject& o2) {
    double overlappingWidth = fmin(o1.x + o1.width, o2.x + o2.width) - fmax(o1.x, o2.x);
    double overlappingHeight = fmin(o1.y + o1.height, o2.y + o2.height) - fmax(o1.y, o2.y);
    double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
    double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
    return intersectionArea / unionArea;
}

//图像缩放与填充
cv::Mat ModelYolo5::letterBox(cv::Mat src) {

	if (src.empty()) { std::cout << "input image invalid" << std::endl;  return cv::Mat(); }
	//以下为带边框图像生成
	int in_w = src.cols;
	int in_h = src.rows;
	int tar_w = 640;
	int tar_h = 640;
	//哪个缩放比例小选哪个
	this->ratio = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
	int inside_w = std::round(in_w * ratio);
	int inside_h = std::round(in_h * ratio);
	int pad_w = tar_w - inside_w;
	int pad_h = tar_h - inside_h;
	//内层图像resize
	cv::Mat resize_img;
	cv::resize(src, resize_img, cv::Size(inside_w, inside_h));  //最小的Resize
	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
	pad_w = pad_w / 2;
	pad_h = pad_h / 2;
	//外层边框填充灰色
	this->topPad = int(std::round(pad_h - 0.1));
	auto btmPad = int(std::round(pad_h + 0.1));
	this->leftPad = int(std::round(pad_w - 0.1));
	auto rightPad = int(std::round(pad_w + 0.1));

	cv::copyMakeBorder(resize_img, resize_img, topPad, btmPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	return resize_img;
}


//还原
//从detect得到的xywh转换回到原图xywh
cv::Rect ModelYolo5::detect2origin(const cv::Rect &det_rect, float rate_to, int top, int left) {
	//detect坐标转换到内部纯图坐标
	int inside_x = det_rect.x - left;
	int inside_y = det_rect.y - top;
	int ox = std::round(float(inside_x) / rate_to);
	int oy = std::round(float(inside_y) / rate_to);
	int ow = std::round(float(det_rect.width) / rate_to);
	int oh = std::round(float(det_rect.height) / rate_to);

	cv::Rect origin_rect(ox, oy, ow, oh);
	return origin_rect;
}


// ModelYolo5::Region::Region(const std::shared_ptr<ngraph::op::RegionYolo>& regionYolo) {
//     coords = regionYolo->get_num_coords();
//     classes = regionYolo->get_num_classes();
//     auto mask = regionYolo->get_mask();
//     num = mask.size();
//     anchors.resize(num * 2);

//     for (int i = 0; i < num; ++i) {
//         anchors[i * 2] = regionYolo->get_anchors()[mask[i] * 2];
//         anchors[i * 2 + 1] = regionYolo->get_anchors()[mask[i] * 2 + 1];
//     }
// }
