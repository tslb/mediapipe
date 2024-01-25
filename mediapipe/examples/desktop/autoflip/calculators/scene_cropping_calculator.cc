// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/examples/desktop/autoflip/calculators/scene_cropping_calculator.h"

#include <cmath>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/scene_cropping_viz.h"
#include "mediapipe/examples/desktop/autoflip/quality/utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace autoflip {

constexpr char kInputVideoFrames[] = "VIDEO_FRAMES";
constexpr char kInputVideoSize[] = "VIDEO_SIZE";
constexpr char kInputKeyFrames[] = "KEY_FRAMES";
constexpr char kInputDetections[] = "DETECTION_FEATURES";
constexpr char kInputStaticFeatures[] = "STATIC_FEATURES";
constexpr char kInputShotBoundaries[] = "SHOT_BOUNDARIES";
constexpr char kInputExternalSettings[] = "EXTERNAL_SETTINGS";
// This side packet must be used in conjunction with
// TargetSizeType::MAXIMIZE_TARGET_DIMENSION
constexpr char kAspectRatio[] = "EXTERNAL_ASPECT_RATIO";

// Output the cropped frames, as well as visualization of crop regions and focus
// points. Note that, KEY_FRAME_CROP_REGION_VIZ_FRAMES and
// SALIENT_POINT_FRAME_VIZ_FRAMES can only be enabled when CROPPED_FRAMES is
// enabled.
constexpr char kOutputCroppedFrames[] = "CROPPED_FRAMES";
// Shows detections on key frames.  Any static borders will be removed from the
// output frame.
constexpr char kOutputKeyFrameCropViz[] = "KEY_FRAME_CROP_REGION_VIZ_FRAMES";
// Shows x/y (raw unsmoothed) cropping and focus points.  Any static borders
// will be removed from the output frame.
constexpr char kOutputFocusPointFrameViz[] = "SALIENT_POINT_FRAME_VIZ_FRAMES";
// Shows final smoothed cropping and a focused area of the camera.  Any static
// borders will remain and be shown in grey.  Output frame will match input
// frame size.
constexpr char kOutputFramingAndDetections[] = "FRAMING_DETECTIONS_VIZ_FRAMES";
// Final summary of cropping.
constexpr char kOutputSummary[] = "CROPPING_SUMMARY";



absl::Status SceneCroppingCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  if (cc->InputSidePackets().HasTag(kInputExternalSettings)) {
    cc->InputSidePackets().Tag(kInputExternalSettings).Set<std::string>();
  }
  if (cc->InputSidePackets().HasTag(kAspectRatio)) {
    cc->InputSidePackets().Tag(kAspectRatio).Set<std::string>();
  }
  if (cc->InputSidePackets().HasTag("OUTPUT_FILE_PATH")){
    cc->InputSidePackets().Tag("OUTPUT_FILE_PATH").Set<std::string>();
  }
  if (cc->Inputs().HasTag(kInputVideoFrames)) {
    cc->Inputs().Tag(kInputVideoFrames).Set<ImageFrame>();
  }
  if (cc->Inputs().HasTag(kInputVideoSize)) {
    cc->Inputs().Tag(kInputVideoSize).Set<std::pair<int, int>>();
  }
  if (cc->Inputs().HasTag(kInputKeyFrames)) {
    cc->Inputs().Tag(kInputKeyFrames).Set<ImageFrame>();
  }
  cc->Inputs().Tag(kInputDetections).Set<DetectionSet>();
  if (cc->Inputs().HasTag(kInputStaticFeatures)) {
    cc->Inputs().Tag(kInputStaticFeatures).Set<StaticFeatures>();
  }
  if (cc->Inputs().HasTag(kInputShotBoundaries)) {
    cc->Inputs().Tag(kInputShotBoundaries).Set<bool>();
  }

  if (cc->Outputs().HasTag(kOutputCroppedFrames)) {
    cc->Outputs().Tag(kOutputCroppedFrames).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputKeyFrameCropViz)) {
    RET_CHECK(cc->Outputs().HasTag(kOutputCroppedFrames))
        << "KEY_FRAME_CROP_REGION_VIZ_FRAMES can only be used when "
           "CROPPED_FRAMES is specified.";
    cc->Outputs().Tag(kOutputKeyFrameCropViz).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputFramingAndDetections)) {
    RET_CHECK(cc->Outputs().HasTag(kOutputCroppedFrames))
        << "FRAMING_DETECTIONS_VIZ_FRAMES can only be used when "
           "CROPPED_FRAMES is specified.";
    cc->Outputs().Tag(kOutputFramingAndDetections).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputFocusPointFrameViz)) {
    RET_CHECK(cc->Outputs().HasTag(kOutputCroppedFrames))
        << "SALIENT_POINT_FRAME_VIZ_FRAMES can only be used when "
           "CROPPED_FRAMES is specified.";
    cc->Outputs().Tag(kOutputFocusPointFrameViz).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputSummary)) {
    cc->Outputs().Tag(kOutputSummary).Set<VideoCroppingSummary>();
  }

  RET_CHECK(cc->Inputs().HasTag(kInputVideoFrames) ^
            cc->Inputs().HasTag(kInputVideoSize))
      << "VIDEO_FRAMES or VIDEO_SIZE must be set and not both.";
  RET_CHECK(!(cc->Inputs().HasTag(kInputVideoSize) &&
              cc->Inputs().HasTag(kOutputCroppedFrames)))
      << "CROPPED_FRAMES (internal cropping) has been set as an output without "
         "VIDEO_FRAMES (video data) input.";
  // RET_CHECK(cc->Outputs().HasTag(kOutputCroppedFrames))
  //     << "At leaset one output stream must be specified";
  return absl::OkStatus();
}

absl::Status SceneCroppingCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<SceneCroppingCalculatorOptions>();
  RET_CHECK_GT(options_.max_scene_size(), 0)
      << "Maximum scene size is non-positive.";
  RET_CHECK_GE(options_.prior_frame_buffer_size(), 0)
      << "Prior frame buffer size is negative.";

  RET_CHECK(options_.solid_background_frames_padding_fraction() >= 0.0 &&
            options_.solid_background_frames_padding_fraction() <= 1.0)
      << "Solid background frames padding fraction is not in [0, 1].";
  const auto& padding_params = options_.padding_parameters();
  background_contrast_ = padding_params.background_contrast();
  RET_CHECK(background_contrast_ >= 0.0 && background_contrast_ <= 1.0)
      << "Background contrast " << background_contrast_ << " is not in [0, 1].";
  blur_cv_size_ = padding_params.blur_cv_size();
  RET_CHECK_GT(blur_cv_size_, 0) << "Blur cv size is non-positive.";
  overlay_opacity_ = padding_params.overlay_opacity();
  RET_CHECK(overlay_opacity_ >= 0.0 && overlay_opacity_ <= 1.0)
      << "Overlay opacity " << overlay_opacity_ << " is not in [0, 1].";

  // Set default camera model to polynomial_path_solver.
  if (!options_.camera_motion_options().has_kinematic_options()) {
    options_.mutable_camera_motion_options()
        ->mutable_polynomial_path_solver()
        ->set_prior_frame_buffer_size(options_.prior_frame_buffer_size());
  }
  if (cc->Outputs().HasTag(kOutputSummary)) {
    summary_ = absl::make_unique<VideoCroppingSummary>();
  }
  
  // handles output file details, removing it if it already exists
  output_file_path_ =
      cc->InputSidePackets().Tag("OUTPUT_FILE_PATH").Get<std::string>();
  remove(output_file_path_.c_str());  

  should_perform_frame_cropping_ = cc->Outputs().HasTag(kOutputCroppedFrames);
  scene_camera_motion_analyzer_ = absl::make_unique<SceneCameraMotionAnalyzer>(
      options_.scene_camera_motion_analyzer_options());
  return absl::OkStatus();
}

namespace {
absl::Status ParseAspectRatioString(const std::string& aspect_ratio_string,
                                    double* aspect_ratio) {
  std::string error_msg =
      "Aspect ratio string must be in the format of 'width:height', e.g. "
      "'1:1' or '5:4', your input was " +
      aspect_ratio_string;
  auto pos = aspect_ratio_string.find(':');
  RET_CHECK(pos != std::string::npos) << error_msg;
  double width_ratio;
  RET_CHECK(absl::SimpleAtod(aspect_ratio_string.substr(0, pos), &width_ratio))
      << error_msg;
  double height_ratio;
  RET_CHECK(absl::SimpleAtod(
      aspect_ratio_string.substr(pos + 1, aspect_ratio_string.size()),
      &height_ratio))
      << error_msg;
  *aspect_ratio = width_ratio / height_ratio;
  return absl::OkStatus();
}

double GetRatio(int width, int height) {
  return static_cast<double>(width) / height;
}

int RoundToEven(float value) {
  int rounded_value = std::round(value);
  if (rounded_value % 2 == 1) {
    rounded_value = std::max(2, rounded_value - 1);
  }
  return rounded_value;
}

}  // namespace

absl::Status SceneCroppingCalculator::InitializeSceneCroppingCalculator(
    mediapipe::CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kInputVideoFrames)) {
    const auto& frame = cc->Inputs().Tag(kInputVideoFrames).Get<ImageFrame>();
    frame_width_ = frame.Width();
    frame_height_ = frame.Height();
    frame_format_ = frame.Format();
  } else if (cc->Inputs().HasTag(kInputVideoSize)) {
    frame_width_ =
        cc->Inputs().Tag(kInputVideoSize).Get<std::pair<int, int>>().first;
    frame_height_ =
        cc->Inputs().Tag(kInputVideoSize).Get<std::pair<int, int>>().second;
  } else {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Input VIDEO or VIDEO_SIZE must be provided.";
  }
  RET_CHECK_GT(frame_height_, 0) << "Input frame height is non-positive.";
  RET_CHECK_GT(frame_width_, 0) << "Input frame width is non-positive.";

  // Calculate target width and height.
  switch (options_.target_size_type()) {
    case SceneCroppingCalculatorOptions::KEEP_ORIGINAL_HEIGHT:
      RET_CHECK(options_.has_target_width() && options_.has_target_height())
          << "Target width and height have to be specified.";
      target_height_ = RoundToEven(frame_height_);
      target_width_ =
          RoundToEven(target_height_ * GetRatio(options_.target_width(),
                                                options_.target_height()));
      break;
    case SceneCroppingCalculatorOptions::KEEP_ORIGINAL_WIDTH:
      RET_CHECK(options_.has_target_width() && options_.has_target_height())
          << "Target width and height have to be specified.";
      target_width_ = RoundToEven(frame_width_);
      target_height_ =
          RoundToEven(target_width_ / GetRatio(options_.target_width(),
                                               options_.target_height()));
      break;
    case SceneCroppingCalculatorOptions::MAXIMIZE_TARGET_DIMENSION: {
      RET_CHECK(cc->InputSidePackets().HasTag(kAspectRatio))
          << "MAXIMIZE_TARGET_DIMENSION is set without an "
             "external_aspect_ratio";
      double requested_aspect_ratio;
      MP_RETURN_IF_ERROR(ParseAspectRatioString(
          cc->InputSidePackets().Tag(kAspectRatio).Get<std::string>(),
          &requested_aspect_ratio));
      const double original_aspect_ratio =
          GetRatio(frame_width_, frame_height_);
      if (original_aspect_ratio > requested_aspect_ratio) {
        target_height_ = RoundToEven(frame_height_);
        target_width_ = RoundToEven(target_height_ * requested_aspect_ratio);
      } else {
        target_width_ = RoundToEven(frame_width_);
        target_height_ = RoundToEven(target_width_ / requested_aspect_ratio);
      }
      break;
    }
    case SceneCroppingCalculatorOptions::USE_TARGET_DIMENSION:
      RET_CHECK(options_.has_target_width() && options_.has_target_height())
          << "Target width and height have to be specified.";
      target_width_ = options_.target_width();
      target_height_ = options_.target_height();
      break;
    case SceneCroppingCalculatorOptions::KEEP_ORIGINAL_DIMENSION:
      target_width_ = frame_width_;
      target_height_ = frame_height_;
      break;
    case SceneCroppingCalculatorOptions::UNKNOWN:
      return absl::InvalidArgumentError("target_size_type not set properly.");
  }
  target_aspect_ratio_ = GetRatio(target_width_, target_height_);

  // Set keyframe width/height for feature upscaling.
  RET_CHECK(!(cc->Inputs().HasTag(kInputKeyFrames) &&
              (options_.has_video_features_width() ||
               options_.has_video_features_height())))
      << "Key frame size must be defined by either providing the input stream "
         "KEY_FRAMES or setting video_features_width/video_features_height as "
         "calculator options.  Both methods cannot be used together.";
  if (options_.has_video_features_width() &&
      options_.has_video_features_height()) {
    key_frame_width_ = options_.video_features_width();
    key_frame_height_ = options_.video_features_height();
  } else if (!cc->Inputs().HasTag(kInputKeyFrames)) {
    key_frame_width_ = frame_width_;
    key_frame_height_ = frame_height_;
  }
  // Check provided dimensions.
  RET_CHECK_GT(target_width_, 0) << "Target width is non-positive.";
  // TODO: it seems this check is too strict and maybe limiting,
  // considering the receiver of frames can be something other than encoder.
  RET_CHECK_NE(target_width_ % 2, 1)
      << "Target width cannot be odd, because encoder expects dimension "
         "values to be even.";
  RET_CHECK_GT(target_height_, 0) << "Target height is non-positive.";
  RET_CHECK_NE(target_height_ % 2, 1)
      << "Target height cannot be odd, because encoder expects dimension "
         "values to be even.";

  scene_cropper_ = absl::make_unique<SceneCropper>(
      options_.camera_motion_options(), frame_width_, frame_height_);

  return absl::OkStatus();
}

bool HasFrameSignal(mediapipe::CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kInputVideoFrames)) {
    return !cc->Inputs().Tag(kInputVideoFrames).Value().IsEmpty();
  }
  return !cc->Inputs().Tag(kInputVideoSize).Value().IsEmpty();
}

absl::Status SceneCroppingCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  // Sets frame dimension and initializes scenecroppingcalculator on first video
  // frame.
  if (frame_width_ < 0) {
    MP_RETURN_IF_ERROR(InitializeSceneCroppingCalculator(cc));
  }

  // Sets key frame dimension on first keyframe.
  if (cc->Inputs().HasTag(kInputKeyFrames) &&
      !cc->Inputs().Tag(kInputKeyFrames).Value().IsEmpty() &&
      key_frame_width_ < 0) {
    const auto& key_frame = cc->Inputs().Tag(kInputKeyFrames).Get<ImageFrame>();
    key_frame_width_ = key_frame.Width();
    key_frame_height_ = key_frame.Height();
  }

  // Processes a scene when shot boundary or buffer is full.
  bool is_end_of_scene = false;
  if (cc->Inputs().HasTag(kInputShotBoundaries) &&
      !cc->Inputs().Tag(kInputShotBoundaries).Value().IsEmpty()) {
    is_end_of_scene = cc->Inputs().Tag(kInputShotBoundaries).Get<bool>();
  }

  if (!scene_frame_timestamps_.empty() && (is_end_of_scene)) {
    continue_last_scene_ = false;
    MP_RETURN_IF_ERROR(ProcessScene(is_end_of_scene, cc));
  }
  // save memory
  should_perform_frame_cropping_ = false;

  // Saves frame and timestamp and whether it is a key frame.
  if (HasFrameSignal(cc)) {
    // Only buffer frames if |should_perform_frame_cropping_| is true.
    if (should_perform_frame_cropping_) {
      const auto& frame = cc->Inputs().Tag(kInputVideoFrames).Get<ImageFrame>();
      const cv::Mat frame_mat = formats::MatView(&frame);
      cv::Mat copy_mat;
      frame_mat.copyTo(copy_mat);
      scene_frames_or_empty_.push_back(copy_mat);
    }
    scene_frame_timestamps_.push_back(cc->InputTimestamp().Value());
    is_key_frames_.push_back(
        !cc->Inputs().Tag(kInputDetections).Value().IsEmpty());
  }

  // Packs key frame info.
  if (!cc->Inputs().Tag(kInputDetections).Value().IsEmpty()) {
    const auto& detections =
        cc->Inputs().Tag(kInputDetections).Get<DetectionSet>();
    KeyFrameInfo key_frame_info;
    MP_RETURN_IF_ERROR(PackKeyFrameInfo(
        cc->InputTimestamp().Value(), detections, frame_width_, frame_height_,
        key_frame_width_, key_frame_height_, &key_frame_info));
    key_frame_infos_.push_back(key_frame_info);
  }

  // Buffers static features.
  if (cc->Inputs().HasTag(kInputStaticFeatures) &&
      !cc->Inputs().Tag(kInputStaticFeatures).Value().IsEmpty()) {
    static_features_.push_back(
        cc->Inputs().Tag(kInputStaticFeatures).Get<StaticFeatures>());
    static_features_timestamps_.push_back(cc->InputTimestamp().Value());
  }

  const bool force_buffer_flush =
      scene_frame_timestamps_.size() >= options_.max_scene_size();
  if (!scene_frame_timestamps_.empty() && force_buffer_flush) {
    MP_RETURN_IF_ERROR(ProcessScene(is_end_of_scene, cc));
    continue_last_scene_ = true;
  }

  return absl::OkStatus();
}

absl::Status SceneCroppingCalculator::Close(mediapipe::CalculatorContext* cc) {
  if (!scene_frame_timestamps_.empty()) {
    MP_RETURN_IF_ERROR(ProcessScene(/* is_end_of_scene = */ true, cc));
  }
  if (cc->Outputs().HasTag(kOutputSummary)) {
    cc->Outputs()
        .Tag(kOutputSummary)
        .Add(summary_.release(), Timestamp::PostStream());
  }
  return absl::OkStatus();
}

// TODO: split this function into two, one for calculating the border
// sizes, the other for the actual removal of borders from the frames.
absl::Status SceneCroppingCalculator::RemoveStaticBorders(
    CalculatorContext* cc, int* top_border_size, int* bottom_border_size) {
  *top_border_size = 0;
  *bottom_border_size = 0;
  MP_RETURN_IF_ERROR(ComputeSceneStaticBordersSize(
      static_features_, top_border_size, bottom_border_size));
  const double scale = static_cast<double>(frame_height_) / key_frame_height_;
  top_border_distance_ = std::round(scale * *top_border_size);
  const int bottom_border_distance = std::round(scale * *bottom_border_size);
  effective_frame_height_ = frame_height_;

  // Store shallow copy of the original frames for debug display if required
  // before static areas are removed.
  if (cc->Outputs().HasTag(kOutputFramingAndDetections)) {
    raw_scene_frames_or_empty_ = {scene_frames_or_empty_.begin(),
                                  scene_frames_or_empty_.end()};
  }

  if (top_border_distance_ > 0 || bottom_border_distance > 0) {
    VLOG(1) << "Remove top border " << top_border_distance_ << " bottom border "
            << bottom_border_distance;
    // Remove borders from frames.
    cv::Rect roi(0, top_border_distance_, frame_width_,
                 effective_frame_height_);
    for (int i = 0; i < scene_frames_or_empty_.size(); ++i) {
      cv::Mat tmp;
      scene_frames_or_empty_[i](roi).copyTo(tmp);
      scene_frames_or_empty_[i] = tmp;
    }
    // Adjust detection bounding boxes.
    for (int i = 0; i < key_frame_infos_.size(); ++i) {
      DetectionSet adjusted_detections;
      const auto& detections = key_frame_infos_[i].detections();
      for (int j = 0; j < detections.detections_size(); ++j) {
        const auto& detection = detections.detections(j);
        SalientRegion adjusted_detection = detection;
        // Clamp the box to be within the de-bordered frame.
        if (!ClampRect(0, top_border_distance_, frame_width_,
                       top_border_distance_ + effective_frame_height_,
                       adjusted_detection.mutable_location())
                 .ok()) {
          continue;
        }
        // Offset the y position.
        adjusted_detection.mutable_location()->set_y(
            adjusted_detection.location().y() - top_border_distance_);
        *adjusted_detections.add_detections() = adjusted_detection;
      }
      *key_frame_infos_[i].mutable_detections() = adjusted_detections;
    }
  }
  return absl::OkStatus();
}

absl::Status SceneCroppingCalculator::InitializeFrameCropRegionComputer() {
  key_frame_crop_options_ = options_.key_frame_crop_options();
  MP_RETURN_IF_ERROR(
      SetKeyFrameCropTarget(frame_width_, effective_frame_height_,
                            target_aspect_ratio_, &key_frame_crop_options_));
  VLOG(1) << "Target width " << key_frame_crop_options_.target_width();
  VLOG(1) << "Target height " << key_frame_crop_options_.target_height();
  frame_crop_region_computer_ =
      absl::make_unique<FrameCropRegionComputer>(key_frame_crop_options_);
  return absl::OkStatus();
}

void SceneCroppingCalculator::FilterKeyFrameInfo() {
  if (!options_.user_hint_override()) {
    return;
  }
  std::vector<KeyFrameInfo> user_hints_only;
  bool has_user_hints = false;
  for (auto key_frame : key_frame_infos_) {
    DetectionSet user_hint_only_set;
    for (const auto& detection : key_frame.detections().detections()) {
      if (detection.signal_type().has_standard() &&
          detection.signal_type().standard() == SignalType::USER_HINT) {
        *user_hint_only_set.add_detections() = detection;
        has_user_hints = true;
      }
    }
    *key_frame.mutable_detections() = user_hint_only_set;
    user_hints_only.push_back(key_frame);
  }
  if (has_user_hints) {
    key_frame_infos_ = user_hints_only;
  }
}

absl::Status SceneCroppingCalculator::ProcessScene(const bool is_end_of_scene,
                                                   CalculatorContext* cc) {
  // Removes detections under special circumstances.
  FilterKeyFrameInfo();

  // Removes any static borders.
  int top_static_border_size, bottom_static_border_size;
  MP_RETURN_IF_ERROR(RemoveStaticBorders(cc, &top_static_border_size,
                                         &bottom_static_border_size));

  // Decides if solid background color padding is possible and sets up color
  // interpolation functions in CIELAB. Uses linear interpolation by default.
  MP_RETURN_IF_ERROR(FindSolidBackgroundColor(
      static_features_, static_features_timestamps_,
      options_.solid_background_frames_padding_fraction(),
      &has_solid_background_, &background_color_l_function_,
      &background_color_a_function_, &background_color_b_function_));

  // Computes key frame crop regions and moves information from raw
  // key_frame_infos_ to key_frame_crop_results.
  MP_RETURN_IF_ERROR(InitializeFrameCropRegionComputer());
  const int num_key_frames = key_frame_infos_.size();
  std::vector<KeyFrameCropResult> key_frame_crop_results(num_key_frames);
  for (int i = 0; i < num_key_frames; ++i) {
    MP_RETURN_IF_ERROR(frame_crop_region_computer_->ComputeFrameCropRegion(
        key_frame_infos_[i], &key_frame_crop_results[i]));
  }

  SceneKeyFrameCropSummary scene_summary;
  std::vector<FocusPointFrame> focus_point_frames;
  SceneCameraMotion scene_camera_motion;
  MP_RETURN_IF_ERROR(
      scene_camera_motion_analyzer_->AnalyzeSceneAndPopulateFocusPointFrames(
          key_frame_crop_options_, key_frame_crop_results, frame_width_,
          effective_frame_height_, scene_frame_timestamps_,
          has_solid_background_, &scene_summary, &focus_point_frames,
          &scene_camera_motion));

  // Crops scene frames.
  std::vector<cv::Mat> cropped_frames;
  std::vector<cv::Rect> crop_from_locations;
  
  auto* cropped_frames_ptr =
      should_perform_frame_cropping_ ? &cropped_frames : nullptr;

  MP_RETURN_IF_ERROR(scene_cropper_->CropFrames(
      scene_summary, scene_frame_timestamps_, is_key_frames_,
      scene_frames_or_empty_, focus_point_frames, prior_focus_point_frames_,
      top_static_border_size, bottom_static_border_size, continue_last_scene_,
      &crop_from_locations, cropped_frames_ptr, output_file_path_));

  const double start_sec = Timestamp(scene_frame_timestamps_.front()).Seconds();
  const double end_sec = Timestamp(scene_frame_timestamps_.back()).Seconds();
  VLOG(1) << absl::StrFormat("Processed a scene from %.2f sec to %.2f sec",
                             start_sec, end_sec);

  key_frame_infos_.clear();
  scene_frames_or_empty_.clear();
  scene_frame_timestamps_.clear();
  is_key_frames_.clear();
  static_features_.clear();
  static_features_timestamps_.clear();
  return absl::OkStatus();
}



REGISTER_CALCULATOR(SceneCroppingCalculator);

}  // namespace autoflip
}  // namespace mediapipe
