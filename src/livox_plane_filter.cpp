#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <deque>
#include <vector>
#include <random>
#include <algorithm>
#include <mutex>

struct Chunk {
  std::vector<float> xyz; // flat: x0,y0,z0,x1,y1,z1,...
  ros::Time stamp;
};

class PlaneFilterNode {
public:
  PlaneFilterNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh) {

    // Params
    pnh_.param<std::string>("input_topic", input_topic_, std::string("/livox/lidar"));
    pnh_.param<std::string>("frame_id", frame_id_, std::string("livox_frame"));
    pnh_.param("publish_rate_hz", publish_rate_hz_, 5.0);
    pnh_.param("max_points_per_publish", max_points_per_publish_, 120000);
    pnh_.param("window_sec", window_sec_, 0.0);

    pnh_.param("angle_deg", angle_deg_, -11.0);
    pnh_.param("thickness", thickness_, 0.1);
    pnh_.param("min_range", min_range_, 0.3);
    pnh_.param("max_range", max_range_, 15.0);
    pnh_.param("front_only", front_only_, true);

    int decimation = 15;
    pnh_.param("decimation", decimation, decimation);
    pnh_.param("ring_len", ring_len_, 200000);
    pnh_.param("seed", seed_, 42);

    pnh_.param("keep_prob", keep_prob_, -1.0);
    if (keep_prob_ < 0.0) keep_prob_ = 1.0 / std::max(1, decimation);
    keep_prob_ = std::max(0.0, std::min(1.0, keep_prob_));

    // Precompute plane and ranges
    const double theta = angle_deg_ * M_PI / 180.0;
    nx_ = std::sin(theta);
    nz_ = std::cos(theta);
    half_thickness_ = thickness_ * 0.5;
    min_r2_ = min_range_ * min_range_;
    max_r2_ = max_range_ * max_range_;

    // Build random ring mask
    buildRingMask();

    pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("filtered", 1);
    sub_ = nh_.subscribe(input_topic_, 1, &PlaneFilterNode::lidarCb, this);

    const double period = 1.0 / std::max(1e-6, publish_rate_hz_);
    timer_ = nh_.createTimer(ros::Duration(period), &PlaneFilterNode::onTimer, this);

    ROS_INFO_STREAM("mid360_avoidance: rate=" << publish_rate_hz_
                    << " Hz, keep_prob=" << keep_prob_
                    << ", ring_len=" << ring_len_
                    << ", window=" << window_sec_ << " s");
  }

private:
  void buildRingMask() {
    ring_mask_.assign(ring_len_, 0u);
    const int n_true = static_cast<int>(std::round(ring_len_ * keep_prob_));
    for (int i = 0; i < n_true && i < ring_len_; ++i) ring_mask_[i] = 1u;
    std::mt19937 rng(seed_);
    std::shuffle(ring_mask_.begin(), ring_mask_.end(), rng);
    ring_pos_ = 0;
  }

  inline uint8_t ringValueAdvance() {
    // return current, then advance
    const uint8_t v = ring_mask_[ring_pos_];
    ring_pos_++;
    if (ring_pos_ >= ring_len_) ring_pos_ = 0;
    return v;
  }

  void pruneWindowLocked(const ros::Time& now) {
    if (window_sec_ <= 0.0) return;
    const ros::Time cutoff = now - ros::Duration(window_sec_);
    while (!chunks_.empty() && chunks_.front().stamp < cutoff) {
      chunks_.pop_front();
    }
  }

  void lidarCb(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    const size_t n = msg->points.size();
    if (n == 0) return;

    // Reserve with rough estimate to reduce reallocs
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(std::ceil(n * keep_prob_)) * 3);

    // 1) Subsample by ring mask + 2) front + 3) corridor + range; 4) project z=0
    for (size_t i = 0; i < n; ++i) {
      if (!ringValueAdvance()) continue; // skip

      const auto& p = msg->points[i];
      const float x = p.x;
      const float y = p.y;
      const float z = p.z;

      if (front_only_ && x < 0.0f) continue;

      const float dist = x * nx_ + z * nz_;
      if (std::fabs(dist) >= half_thickness_) continue;

      const float r2 = x * x + y * y + z * z;
      if (r2 < min_r2_ || r2 > max_r2_) continue;

      // project to z=0
      flat.push_back(x);
      flat.push_back(y);
      flat.push_back(0.0f);
    }

    if (flat.empty()) return;

    const ros::Time now = ros::Time::now();
    std::lock_guard<std::mutex> lk(mtx_);
    chunks_.push_back(Chunk{std::move(flat), now});
    pruneWindowLocked(now);
  }

  void onTimer(const ros::TimerEvent&) {
    std::deque<Chunk> local;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (chunks_.empty()) return;
      const ros::Time now = ros::Time::now();
      pruneWindowLocked(now);
      if (chunks_.empty()) return;

      // Move (steal) for minimal copies
      local = std::move(chunks_);
      if (window_sec_ <= 0.0) {
        chunks_.clear();
      } else {
        // keep (we moved; put back)
        chunks_ = local;
      }
    }

    // total points
    size_t total3 = 0;
    for (const auto& c : local) total3 += c.xyz.size();
    if (total3 == 0) return;

    // optional safety downsample: stride to fit max_points_per_publish_
    size_t total_pts = total3 / 3;
    size_t step = 1;
    if (static_cast<int>(total_pts) > max_points_per_publish_) {
      step = static_cast<size_t>(
          std::ceil(static_cast<double>(total_pts) / static_cast<double>(max_points_per_publish_)));
    }

    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp = ros::Time::now();
    cloud.header.frame_id = frame_id_;
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = false;

    const size_t out_pts = (step == 1) ? total_pts : (total_pts + step - 1) / step;
    cloud.width = static_cast<uint32_t>(out_pts);

    cloud.fields.resize(3);
    cloud.fields[0].name = "x"; cloud.fields[0].offset = 0;  cloud.fields[0].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[0].count = 1;
    cloud.fields[1].name = "y"; cloud.fields[1].offset = 4;  cloud.fields[1].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[1].count = 1;
    cloud.fields[2].name = "z"; cloud.fields[2].offset = 8;  cloud.fields[2].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[2].count = 1;

    cloud.point_step = 12;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);

    sensor_msgs::PointCloud2Iterator<float> it_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> it_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> it_z(cloud, "z");

    size_t skip = 0;
    for (const auto& c : local) {
      const auto& v = c.xyz;
      for (size_t i = 0; i + 2 < v.size(); i += 3) {
        if (step > 1) {
          if (skip++ % step) continue;
        }
        *it_x = v[i + 0];
        *it_y = v[i + 1];
        *it_z = v[i + 2]; // already 0
        ++it_x; ++it_y; ++it_z;
      }
    }

    pub_.publish(cloud);
  }

private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Timer timer_;

  // params
  std::string input_topic_, frame_id_;
  double publish_rate_hz_{5.0};
  int max_points_per_publish_{120000};
  double window_sec_{0.0};

  double angle_deg_{-11.0}, thickness_{0.1}, min_range_{0.3}, max_range_{15.0};
  bool front_only_{true};

  // plane & ranges
  double nx_{0.0}, nz_{1.0}, half_thickness_{0.05};
  double min_r2_{0.09}, max_r2_{225.0};

  // ring mask
  int ring_len_{200000}, seed_{42};
  double keep_prob_{1.0/15.0};
  std::vector<uint8_t> ring_mask_;
  int ring_pos_{0};

  // accumulation
  std::deque<Chunk> chunks_;
  std::mutex mtx_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "plane_filter");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  PlaneFilterNode node(nh, pnh);
  ros::spin();
  return 0;
}
