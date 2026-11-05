/*
 * ESP32-CAM 石头剪刀布识别
 * 模型: model.h (96x96x3, int8, 3 类: paper/rock/scissors)
 * 依赖库: TensorFlowLite_ESP32 (tanakamasayuki)
 * 开发板: AI Thinker ESP32-CAM, 需开启 PSRAM
 */
#include "esp_camera.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// AI Thinker ESP32-CAM 引脚
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_GPIO_NUM     4

constexpr int kImgSize = 96;
constexpr int kArenaSize = 420 * 1024;   // 激活值最大约 96*96*32 字节
uint8_t *tensor_arena = nullptr;

const char *kLabels[3] = {"paper", "rock", "scissors"};

tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;    // 160x120
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  return esp_camera_init(&config) == ESP_OK;
}

bool initModel() {
  tensor_arena = (uint8_t *)ps_malloc(kArenaSize);
  if (!tensor_arena) {
    Serial.println("PSRAM 分配失败");
    return false;
  }
  const tflite::Model *model = tflite::GetModel(gesture_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("模型版本不匹配");
    return false;
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors 失败");
    return false;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  return true;
}

// RGB565 -> 96x96 RGB888, 中心裁 120x120 再最近邻缩放
void fillInput(const uint8_t *src, int w, int h) {
  const int crop = (w < h) ? w : h;
  const int ox = (w - crop) / 2;
  const int oy = (h - crop) / 2;
  uint8_t *dst = input->data.uint8;
  for (int y = 0; y < kImgSize; y++) {
    int sy = oy + (y * crop) / kImgSize;
    for (int x = 0; x < kImgSize; x++) {
      int sx = ox + (x * crop) / kImgSize;
      int idx = (sy * w + sx) * 2;
      uint16_t px = (src[idx] << 8) | src[idx + 1];
      uint8_t r = ((px >> 11) & 0x1F) * 255 / 31;
      uint8_t g = ((px >> 5) & 0x3F) * 255 / 63;
      uint8_t b = (px & 0x1F) * 255 / 31;
      int o = (y * kImgSize + x) * 3;
      dst[o] = r;
      dst[o + 1] = g;
      dst[o + 2] = b;
    }
  }
}

void softmax(const int8_t *logits, float *probs, int n) {
  float maxv = -1e9f;
  for (int i = 0; i < n; i++) maxv = fmaxf(maxv, (float)logits[i] * output->params.scale + output->params.zero_point);
  float sum = 0;
  for (int i = 0; i < n; i++) {
    float v = (float)logits[i] * output->params.scale + output->params.zero_point;
    probs[i] = expf(v - maxv);
    sum += probs[i];
  }
  for (int i = 0; i < n; i++) probs[i] /= sum;
}

void setup() {
  Serial.begin(115200);
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);

  psramInit();
  if (!initCamera()) {
    Serial.println("摄像头初始化失败");
    return;
  }
  if (!initModel()) {
    Serial.println("模型初始化失败");
    return;
  }
  Serial.println("就绪：把手势对准镜头");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("拍照失败");
    delay(500);
    return;
  }
  fillInput(fb->buf, fb->width, fb->height);
  esp_camera_fb_return(fb);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("推理失败");
    delay(500);
    return;
  }

  float probs[3];
  softmax(output->data.int8, probs, 3);
  int best = 0;
  for (int i = 1; i < 3; i++) if (probs[i] > probs[best]) best = i;

  Serial.printf("%s  paper=%.2f rock=%.2f scissors=%.2f\n",
                kLabels[best], probs[0], probs[1], probs[2]);
  delay(300);
}
