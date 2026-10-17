// SerialCam: ESP32-CAM (AI Thinker) streaming JPEG frames over the USB-serial adapter instead of WiFi.
// No WiFi, so no dropouts and much less current (fewer brownouts). 1 Mbaud is about 100 KB/s:
// SVGA at JPEG quality 20 (about 24 KB) gives about 4 fps, VGA 6 to 7. (2 Mbaud lost a few bytes
// in most frames on the CH340 adapter.)
//
// Board -> laptop, packets:  FF AA 55 <type> <uint32 length, little endian> <payload>
//   type 1: one JPEG frame   type 2: settings as JSON (reply to "status")
// Laptop -> board, text lines:  "var=val\n" with the names of the CameraWebServer /control page
//   (framesize, quality, aec, aec_value, agc, agc_gain, awb, awb_gain, lenc, brightness, ...)
//   and "status\n". "led=1\n" / "led=0\n": the white flash LED on GPIO 4 (off at boot).

#include "esp_camera.h"
#include <WiFi.h>

#define BAUD 1000000
#define LED_GPIO_NUM 4  // the flash LED of the AI Thinker board (very bright: it warms the board, so on only when needed)

// AI Thinker ESP32-CAM pins
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

static String line;

static void send_packet(uint8_t type, const uint8_t *data, uint32_t len) {
  uint8_t hdr[8] = {0xFF, 0xAA, 0x55, type,
                    (uint8_t)(len & 0xFF), (uint8_t)((len >> 8) & 0xFF),
                    (uint8_t)((len >> 16) & 0xFF), (uint8_t)((len >> 24) & 0xFF)};
  Serial.write(hdr, 8);
  Serial.write(data, len);
}

static void send_status() {
  sensor_t *s = esp_camera_sensor_get();
  char buf[600];
  int n = snprintf(buf, sizeof(buf),
    "{\"framesize\":%u,\"quality\":%u,\"aec\":%u,\"aec_value\":%u,\"agc\":%u,\"agc_gain\":%u,"
    "\"awb\":%u,\"awb_gain\":%u,\"lenc\":%u,\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
    "\"fs_QVGA\":%d,\"fs_HVGA\":%d,\"fs_VGA\":%d,\"fs_SVGA\":%d,\"fs_XGA\":%d,\"fs_HD\":%d,\"fs_UXGA\":%d,\"led\":%d}",
    s->status.framesize, s->status.quality, s->status.aec, s->status.aec_value, s->status.agc,
    s->status.agc_gain, s->status.awb, s->status.awb_gain, s->status.lenc, s->status.brightness,
    s->status.contrast, s->status.saturation, FRAMESIZE_QVGA, FRAMESIZE_HVGA, FRAMESIZE_VGA, FRAMESIZE_SVGA,
    FRAMESIZE_XGA, FRAMESIZE_HD, FRAMESIZE_UXGA, digitalRead(LED_GPIO_NUM));  // the driver's own numbers (they changed between versions)
  send_packet(2, (const uint8_t *)buf, n);
}

static void handle(const String &cmd) {
  if (cmd == "status") { send_status(); return; }
  int eq = cmd.indexOf('=');
  if (eq < 0) return;
  String var = cmd.substring(0, eq);
  int val = cmd.substring(eq + 1).toInt();
  if (var == "led") { digitalWrite(LED_GPIO_NUM, val ? HIGH : LOW); return; }
  sensor_t *s = esp_camera_sensor_get();
  if (var == "framesize") { if (s->pixformat == PIXFORMAT_JPEG) s->set_framesize(s, (framesize_t)val); }
  else if (var == "quality") s->set_quality(s, val);
  else if (var == "aec") s->set_exposure_ctrl(s, val);
  else if (var == "aec_value") s->set_aec_value(s, val);
  else if (var == "aec2") s->set_aec2(s, val);
  else if (var == "ae_level") s->set_ae_level(s, val);
  else if (var == "agc") s->set_gain_ctrl(s, val);
  else if (var == "agc_gain") s->set_agc_gain(s, val);
  else if (var == "gainceiling") s->set_gainceiling(s, (gainceiling_t)val);
  else if (var == "awb") s->set_whitebal(s, val);
  else if (var == "awb_gain") s->set_awb_gain(s, val);
  else if (var == "wb_mode") s->set_wb_mode(s, val);
  else if (var == "lenc") s->set_lenc(s, val);
  else if (var == "brightness") s->set_brightness(s, val);
  else if (var == "contrast") s->set_contrast(s, val);
  else if (var == "saturation") s->set_saturation(s, val);
  else if (var == "hmirror") s->set_hmirror(s, val);
  else if (var == "vflip") s->set_vflip(s, val);
  else if (var == "bpc") s->set_bpc(s, val);
  else if (var == "wpc") s->set_wpc(s, val);
  else if (var == "raw_gma") s->set_raw_gma(s, val);
  else if (var == "dcw") s->set_dcw(s, val);
}

void setup() {
  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);
  WiFi.mode(WIFI_OFF);  // no radio: no dropouts, less current
  btStop();
  Serial.setRxBufferSize(1024);
  Serial.setTxBufferSize(8192);
  Serial.begin(BAUD);

  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM; c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_SVGA;
  c.jpeg_quality = 20;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;  // always the newest frame: no lag builds up
  if (esp_camera_init(&c) != ESP_OK) {
    const char *msg = "{\"error\":\"camera init failed\"}";
    while (true) { send_packet(2, (const uint8_t *)msg, strlen(msg)); delay(1000); }
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_lenc(s, 1);
}

void loop() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n') { line.trim(); handle(line); line = ""; }
    else if (line.length() < 64) line += ch;
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(5); return; }
  send_packet(1, fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
