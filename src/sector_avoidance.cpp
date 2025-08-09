#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <vector>
#include <algorithm>
#include <cmath>

struct Params {
  // Core
  int    sectors = 9;            // number of angular sectors
  int    k_nearest = 3;          // number of closest points per sector
  double fov_deg = 180.0;        // horizontal field of view (degrees)
  double d_safe = 2.0;           // safe distance threshold (m), beyond this hazard=0
  double beta = 2.0;             // hazard growth exponent
  double max_avoid = 1.0;        // max magnitude of avoidance vector
  double ema_alpha = 0.3;        // EMA smoothing factor for the output vector
  double publish_rate_hz = 20.0; // output publish frequency
  std::string frame_id = "livox_frame";
  std::string input_topic = "/mid360/filtered"; // input cloud topic
  std::string output_topic = "/mid360/avoid_vec";

  // Debug viz (RViz)
  bool   debug_enable = true;        // publish markers
  double debug_scale  = 1.0;         // meters per unit of vector for arrow length
  double debug_alpha  = 0.9;         // alpha for markers
  double debug_lifetime = 0.2;       // seconds
  std::string debug_ns = "sector_avoid";
};

class SectorAvoidance {
public:
  SectorAvoidance(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh) {
    loadParams();
    sub_ = nh_.subscribe(params_.input_topic, 1, &SectorAvoidance::cloudCb, this);
    pub_vec_ = nh_.advertise<geometry_msgs::Vector3Stamped>(params_.output_topic, 1);

    if (params_.debug_enable) {
      pub_markers_ = pnh_.advertise<visualization_msgs::MarkerArray>("debug_markers", 1);
      pub_final_marker_ = pnh_.advertise<visualization_msgs::Marker>("avoid_marker", 1);
    }

    const double period = 1.0 / std::max(1e-6, params_.publish_rate_hz);
    timer_ = nh_.createTimer(ros::Duration(period), &SectorAvoidance::onTimer, this);

    // Precompute FOV parameters
    fov_rad_ = params_.fov_deg * M_PI / 180.0;
    half_fov_ = 0.5 * fov_rad_;
    bin_w_ = fov_rad_ / params_.sectors; // equal angular width per sector
    prev_vx_ = prev_vy_ = 0.0;

    ROS_INFO_STREAM("sector_avoidance: N=" << params_.sectors
                    << " K=" << params_.k_nearest
                    << " fov=" << params_.fov_deg
                    << " d_safe=" << params_.d_safe
                    << " debug=" << (params_.debug_enable ? "on" : "off"));
  }

private:
  void loadParams() {
    // core
    pnh_.param("sectors", params_.sectors, params_.sectors);
    pnh_.param("k_nearest", params_.k_nearest, params_.k_nearest);
    pnh_.param("fov_deg", params_.fov_deg, params_.fov_deg);
    pnh_.param("d_safe", params_.d_safe, params_.d_safe);
    pnh_.param("beta", params_.beta, params_.beta);
    pnh_.param("max_avoid", params_.max_avoid, params_.max_avoid);
    pnh_.param("ema_alpha", params_.ema_alpha, params_.ema_alpha);
    pnh_.param("publish_rate_hz", params_.publish_rate_hz, params_.publish_rate_hz);
    pnh_.param<std::string>("frame_id", params_.frame_id, params_.frame_id);
    pnh_.param<std::string>("input_topic", params_.input_topic, params_.input_topic);
    pnh_.param<std::string>("output_topic", params_.output_topic, params_.output_topic);

    // debug
    pnh_.param("debug_enable", params_.debug_enable, params_.debug_enable);
    pnh_.param("debug_scale", params_.debug_scale, params_.debug_scale);
    pnh_.param("debug_alpha", params_.debug_alpha, params_.debug_alpha);
    pnh_.param("debug_lifetime", params_.debug_lifetime, params_.debug_lifetime);
    pnh_.param<std::string>("debug_ns", params_.debug_ns, params_.debug_ns);

    // bounds
    params_.sectors   = std::max(3, params_.sectors);
    params_.k_nearest = std::max(1, params_.k_nearest);
    params_.d_safe    = std::max(0.05, params_.d_safe);
    params_.beta      = std::max(0.5, params_.beta);
    params_.max_avoid = std::max(0.0, params_.max_avoid);
    params_.ema_alpha = std::min(1.0, std::max(0.0, params_.ema_alpha));
    params_.debug_scale = std::max(1e-3, params_.debug_scale);
    params_.debug_lifetime = std::max(0.0, params_.debug_lifetime);
  }

  void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg) {
    last_frame_id_ = msg->header.frame_id.empty() ? params_.frame_id : msg->header.frame_id;

    // Reset distance bins for each sector
    const int N = params_.sectors;
    if ((int)dist_bins_.size() != N) dist_bins_.assign(N, {});
    else for (auto& v : dist_bins_) v.clear();

    // Iterate through points in the cloud
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    // z is assumed to be zero after preprocessing

    for (; it_x != it_x.end(); ++it_x, ++it_y) {
      const float x = *it_x;
      const float y = *it_y;
      if (x <= 0.0f) continue; // consider only the front hemisphere

      const double ang = std::atan2((double)y, (double)x); // [-pi, pi]
      if (ang < -half_fov_ || ang > half_fov_) continue;

      const int bin = (int)std::floor((ang + half_fov_) / bin_w_);
      if (bin < 0 || bin >= N) continue;

      const double r = std::hypot((double)x, (double)y);
      dist_bins_[bin].push_back(r);
    }
  }

  // Hazard function: clamp((d_safe - d)/d_safe, 0..1)^beta
  inline double hazard(double d) const {
    double x = (params_.d_safe - d) / params_.d_safe;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) x = 1.0;
    return std::pow(x, params_.beta);
  }

  visualization_msgs::Marker makeArrow(
      int id, double sx, double sy, double ex, double ey,
      float r, float g, float b, float a) const
  {
    visualization_msgs::Marker m;
    m.header.stamp = ros::Time::now();
    m.header.frame_id = last_frame_id_.empty() ? params_.frame_id : last_frame_id_;
    m.ns = params_.debug_ns;
    m.id = id;
    m.type = visualization_msgs::Marker::ARROW;
    m.action = visualization_msgs::Marker::ADD;
    m.lifetime = ros::Duration(params_.debug_lifetime);

    // Use points to define arrow start/end (no need for orientation math)
    geometry_msgs::Point p0, p1;
    p0.x = sx; p0.y = sy; p0.z = 0.0;
    p1.x = ex; p1.y = ey; p1.z = 0.0;
    m.points = {p0, p1};

    // Arrow thickness/head
    m.scale.x = 0.03; // shaft diameter (m)
    m.scale.y = 0.06; // head diameter
    m.scale.z = 0.08; // head length

    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    return m;
  }

  void publishDebug(const std::vector<double>& sector_vx,
                    const std::vector<double>& sector_vy,
                    double vx, double vy)
  {
    if (!params_.debug_enable) return;

    visualization_msgs::MarkerArray arr;
    arr.markers.reserve(params_.sectors + 1);

    // Per-sector arrows from origin, length proportional to vector * debug_scale
    for (int i = 0; i < params_.sectors; ++i) {
      const double ex = sector_vx[i] * params_.debug_scale;
      const double ey = sector_vy[i] * params_.debug_scale;

      // Color by magnitude (small -> green, big -> red)
      const double mag = std::hypot(sector_vx[i], sector_vy[i]);
      const double t = std::min(1.0, mag / std::max(1e-6, params_.max_avoid));
      const float r = (float)t;
      const float g = (float)(1.0 - t);
      const float b = 0.1f;

      arr.markers.emplace_back(makeArrow(i, 0.0, 0.0, ex, ey, r, g, b, (float)params_.debug_alpha));
    }

    // Publish per-sector array
    pub_markers_.publish(arr);

    // Publish final vector as separate marker (blue)
    const double ex = vx * params_.debug_scale;
    const double ey = vy * params_.debug_scale;
    auto final_m = makeArrow(1000000, 0.0, 0.0, ex, ey, 0.1f, 0.3f, 1.0f, (float)params_.debug_alpha);
    pub_final_marker_.publish(final_m);
  }

  void onTimer(const ros::TimerEvent&) {
    const int N = params_.sectors;
    if ((int)dist_bins_.size() != N) return;

    double vx = 0.0, vy = 0.0;
    std::vector<double> sector_vx(N, 0.0), sector_vy(N, 0.0);

    for (int i = 0; i < N; ++i) {
      auto& dists = dist_bins_[i];
      if (dists.empty()) continue;

      // Keep only K nearest distances (in-place nth_element)
      const int K = std::min<int>(params_.k_nearest, (int)dists.size());
      std::nth_element(dists.begin(), dists.begin() + K, dists.end());
      dists.resize(K);

      // Compute mean hazard for this sector
      double H = 0.0;
      for (double d : dists) H += hazard(d);
      H /= (double)K;
      if (H <= 0.0) continue;

      // Sector center angle
      const double phi_center = -half_fov_ + (i + 0.5) * bin_w_;
      const double ux = std::cos(phi_center);
      const double uy = std::sin(phi_center);

      // Avoidance vector: opposite to obstacle direction
      const double vxi = -ux * H;
      const double vyi = -uy * H;

      sector_vx[i] = vxi;
      sector_vy[i] = vyi;

      vx += vxi;
      vy += vyi;
    }

    // Limit vector magnitude
    const double norm = std::hypot(vx, vy);
    if (norm > params_.max_avoid && norm > 1e-6) {
      const double s = params_.max_avoid / norm;
      vx *= s; vy *= s;
    }

    // EMA smoothing
    vx = (1.0 - params_.ema_alpha) * prev_vx_ + params_.ema_alpha * vx;
    vy = (1.0 - params_.ema_alpha) * prev_vy_ + params_.ema_alpha * vy;
    prev_vx_ = vx; prev_vy_ = vy;

    // Publish control vector
    geometry_msgs::Vector3Stamped out;
    out.header.stamp = ros::Time::now();
    out.header.frame_id = last_frame_id_.empty() ? params_.frame_id : last_frame_id_;
    out.vector.x = vx;
    out.vector.y = vy;
    out.vector.z = 0.0;
    pub_vec_.publish(out);

    // Publish debug markers (if enabled)
    publishDebug(sector_vx, sector_vy, vx, vy);
  }

private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_;
  ros::Publisher  pub_vec_;
  ros::Publisher  pub_markers_;       // ~debug_markers (MarkerArray)
  ros::Publisher  pub_final_marker_;  // ~avoid_marker (Marker)
  ros::Timer      timer_;

  Params params_;
  double fov_rad_{M_PI}, half_fov_{M_PI_2}, bin_w_{M_PI/9};
  std::vector<std::vector<double>> dist_bins_;
  std::string last_frame_id_;
  double prev_vx_{0.0}, prev_vy_{0.0};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "sector_avoidance");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  SectorAvoidance node(nh, pnh);
  ros::spin();
  return 0;
}
