// Read-only Gazebo classic contact bridge for T3 test instrumentation.
#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
using ContactPair = std::pair<std::string, std::string>;

std::vector<std::string> split_scoped_name(const std::string & value)
{
  std::vector<std::string> parts;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const auto end = value.find("::", begin);
    parts.push_back(value.substr(begin, end - begin));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 2;
  }
  return parts;
}

bool is_go2_trunk_collision(const std::string & value)
{
  const auto parts = split_scoped_name(value);
  return parts.size() >= 3 && parts[0].rfind("go2_", 0) == 0 &&
         (parts[1] == "trunk" ||
          parts[2].find("fixed_joint_lump__trunk_collision") !=
            std::string::npos);
}

std::string json_escape(const std::string & value)
{
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << character;
    }
  }
  return output.str();
}
}  // namespace

class BodyContactMonitor : public rclcpp::Node
{
public:
  BodyContactMonitor()
  : Node("body_contact_monitor")
  {
    publisher_ = create_publisher<std_msgs::msg::String>(
      "/go2_test/body_contacts", rclcpp::QoS(10).reliable());
    gazebo::client::setup();
    gazebo_node_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
    gazebo_node_->Init();
    gazebo_subscription_ = gazebo_node_->Subscribe(
      "~/physics/contacts", &BodyContactMonitor::on_contacts, this);
    timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&BodyContactMonitor::publish_contacts, this));
  }

private:
  void on_contacts(ConstContactsPtr & message)
  {
    std::set<ContactPair> current;
    for (int index = 0; index < message->contact_size(); ++index) {
      std::string first = message->contact(index).collision1();
      std::string second = message->contact(index).collision2();
      if (
        first.find("phase6_test_box") != std::string::npos ||
        second.find("phase6_test_box") != std::string::npos)
      {
        RCLCPP_INFO_ONCE(
          get_logger(), "Observed Phase 6 scoped contact pair: %s <-> %s",
          first.c_str(), second.c_str());
      }
      if (!is_go2_trunk_collision(first) && !is_go2_trunk_collision(second)) {
        continue;
      }
      if (second < first) {
        std::swap(first, second);
      }
      current.emplace(first, second);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    active_contacts_ = std::move(current);
  }

  void publish_contacts()
  {
    std::set<ContactPair> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot = active_contacts_;
    }
    const auto stamp = get_clock()->now();
    const auto stamp_nanoseconds = stamp.nanoseconds();
    std::ostringstream json;
    json << "{\"schema_version\":1,\"stamp\":{\"sec\":"
         << (stamp_nanoseconds / 1000000000LL) << ",\"nanosec\":"
         << (stamp_nanoseconds % 1000000000LL) << "},\"contacts\":[";
    bool first_item = true;
    for (const auto & contact : snapshot) {
      if (!first_item) {
        json << ',';
      }
      first_item = false;
      json << "{\"collision1\":\"" << json_escape(contact.first)
           << "\",\"collision2\":\"" << json_escape(contact.second)
           << "\"}";
    }
    json << "]}";
    std_msgs::msg::String output;
    output.data = json.str();
    publisher_->publish(output);
  }

  std::mutex mutex_;
  std::set<ContactPair> active_contacts_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  gazebo::transport::NodePtr gazebo_node_;
  gazebo::transport::SubscriberPtr gazebo_subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BodyContactMonitor>();
  rclcpp::spin(node);
  gazebo::client::shutdown();
  rclcpp::shutdown();
  return 0;
}
