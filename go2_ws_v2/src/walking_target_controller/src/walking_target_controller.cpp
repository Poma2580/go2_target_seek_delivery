#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <gazebo/common/Animation.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/common/UpdateInfo.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/Actor.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <ignition/math/Pose3.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace walking_target_controller
{

class WalkingTargetController final : public gazebo::ModelPlugin
{
public:
  WalkingTargetController() = default;

  ~WalkingTargetController() override
  {
    if (update_connection_) {
      update_connection_.reset();
    }

    if (executor_) {
      executor_->cancel();
    }
    if (executor_thread_.joinable()) {
      executor_thread_.join();
    }
  }

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    actor_ = boost::dynamic_pointer_cast<gazebo::physics::Actor>(model);
    if (!actor_) {
      gzerr << "walking_target_controller must be attached to a Gazebo actor\n";
      return;
    }

    service_prefix_ = "/walking_target";
    if (sdf && sdf->HasElement("service_prefix")) {
      service_prefix_ = sdf->Get<std::string>("service_prefix");
    }
    if (service_prefix_.empty() || service_prefix_[0] != '/') {
      service_prefix_ = "/" + service_prefix_;
    }
    while (service_prefix_.size() > 1 && service_prefix_.back() == '/') {
      service_prefix_.pop_back();
    }

    if (sdf && sdf->HasElement("animation_factor")) {
      animation_factor_ = sdf->Get<double>("animation_factor");
    }
    if (animation_factor_ <= 0.0) {
      gzerr << "walking_target_controller animation_factor must be positive\n";
      return;
    }

    if (sdf && sdf->HasElement("actor_pose_offset")) {
      actor_pose_offset_ =
        sdf->Get<ignition::math::Pose3d>("actor_pose_offset");
    }

    if (!LoadTrajectory()) {
      return;
    }

    // gazebo_target_seek_world loads libgazebo_ros_init.so before this plugin,
    // so the process-wide ROS 2 context is already initialized. Do not call
    // rclcpp::shutdown() from this plugin: other Gazebo ROS nodes share it.
    node_ = std::make_shared<rclcpp::Node>(
      "walking_target_controller",
      rclcpp::NodeOptions().use_intra_process_comms(false));

    start_service_ = node_->create_service<std_srvs::srv::Trigger>(
      service_prefix_ + "/start",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = true;
        response->success = true;
        response->message = "walking_target started";
      });

    pause_service_ = node_->create_service<std_srvs::srv::Trigger>(
      service_prefix_ + "/pause",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        response->success = true;
        response->message = "walking_target paused";
      });

    reset_service_ = node_->create_service<std_srvs::srv::Trigger>(
      service_prefix_ + "/reset",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        reset_requested_ = true;
        response->success = true;
        response->message = "walking_target reset to trajectory start and paused";
      });

    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    executor_thread_ = std::thread([this]() { executor_->spin(); });

    // A custom trajectory leaves script time under plugin control. Keep the
    // actor active so Gazebo publishes the pose and skeleton frame selected by
    // the controller, even while the controller's route clock is paused.
    actor_->SetCustomTrajectory(custom_trajectory_);
    actor_->SetWorldPose(controlled_pose_, false, false);
    actor_->SetScriptTime(animation_time_);
    actor_->Play();

    // Update before Actor::Update so the model pose and bone animation Gazebo
    // publishes during this simulation tick are derived from the same state.
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&WalkingTargetController::OnUpdate, this, std::placeholders::_1));

    gzlog << "walking_target_controller loaded for actor '" << model->GetName()
          << "'. Services: " << service_prefix_ << "/{start,pause,reset}\n";
  }

private:
  bool LoadTrajectory()
  {
    const auto actor_sdf = actor_->GetSDF();
    if (!actor_sdf || !actor_sdf->HasElement("script")) {
      gzerr << "walking_target_controller actor has no trajectory script\n";
      return false;
    }

    const auto script_sdf = actor_sdf->GetElement("script");
    if (!script_sdf->HasElement("trajectory")) {
      gzerr << "walking_target_controller actor script has no trajectory\n";
      return false;
    }

    const auto trajectory_sdf = script_sdf->GetElement("trajectory");
    if (!trajectory_sdf->HasElement("waypoint")) {
      gzerr << "walking_target_controller trajectory has no waypoints\n";
      return false;
    }

    const auto trajectory_type = trajectory_sdf->Get<std::string>("type");
    const auto skeleton_animations = actor_->SkeletonAnimations();
    if (skeleton_animations.find(trajectory_type) == skeleton_animations.end()) {
      gzerr << "walking_target_controller animation '" << trajectory_type
            << "' was not loaded by the actor\n";
      return false;
    }

    std::map<double, ignition::math::Pose3d> waypoints;
    auto waypoint_sdf = trajectory_sdf->GetElement("waypoint");
    while (waypoint_sdf) {
      waypoints[waypoint_sdf->Get<double>("time")] =
        waypoint_sdf->Get<ignition::math::Pose3d>("pose");
      waypoint_sdf = waypoint_sdf->GetNextElement("waypoint");
    }

    if (waypoints.size() < 2 || waypoints.rbegin()->first <= 0.0) {
      gzerr << "walking_target_controller trajectory needs at least two "
            << "waypoints and a positive duration\n";
      return false;
    }

    route_duration_ = waypoints.rbegin()->first;
    const double tension = trajectory_sdf->Get<double>("tension", 0.0).first;
    route_animation_ = std::make_unique<gazebo::common::PoseAnimation>(
      actor_->GetName() + "_controller_route", route_duration_, true, tension);

    for (auto it = waypoints.begin(); it != waypoints.end(); ++it) {
      const double key_time =
        it == waypoints.begin() ? 0.0 : it->first;
      auto * key = route_animation_->CreateKeyFrame(key_time);
      key->Translation(it->second.Pos());
      key->Rotation(it->second.Rot());
    }

    route_animation_->SetTime(0.0);
    gazebo::common::PoseKeyFrame initial_frame(0.0);
    route_animation_->GetInterpolatedKeyFrame(initial_frame);
    const ignition::math::Pose3d initial_route_pose(
      initial_frame.Translation(), initial_frame.Rotation());
    controlled_pose_ = initial_route_pose * actor_pose_offset_;

    custom_trajectory_ = std::make_shared<gazebo::physics::TrajectoryInfo>();
    custom_trajectory_->id = trajectory_sdf->Get<unsigned int>("id");
    custom_trajectory_->type = trajectory_type;
    custom_trajectory_->duration = route_duration_;
    custom_trajectory_->startTime = 0.0;
    custom_trajectory_->endTime = route_duration_;
    custom_trajectory_->translated = true;
    return true;
  }

  void OnUpdate(const gazebo::common::UpdateInfo & info)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!time_initialized_) {
      last_update_time_ = info.simTime;
      time_initialized_ = true;
    }

    const double dt = std::max(0.0, (info.simTime - last_update_time_).Double());
    last_update_time_ = info.simTime;

    if (reset_requested_) {
      route_time_ = 0.0;
      animation_time_ = 0.0;
      reset_requested_ = false;
    } else if (running_) {
      route_time_ = std::fmod(route_time_ + dt, route_duration_);
    }

    route_animation_->SetTime(route_time_);
    gazebo::common::PoseKeyFrame route_frame(route_time_);
    route_animation_->GetInterpolatedKeyFrame(route_frame);

    const ignition::math::Pose3d route_pose(
      route_frame.Translation(), route_frame.Rotation());
    const ignition::math::Pose3d next_pose = route_pose * actor_pose_offset_;

    if (running_) {
      animation_time_ +=
        controlled_pose_.Pos().Distance(next_pose.Pos()) * animation_factor_;
    }
    controlled_pose_ = next_pose;

    actor_->SetWorldPose(controlled_pose_, false, false);
    actor_->SetScriptTime(animation_time_);
  }

  gazebo::physics::ActorPtr actor_;
  gazebo::event::ConnectionPtr update_connection_;

  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::Executor> executor_;
  std::thread executor_thread_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr pause_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;

  std::unique_ptr<gazebo::common::PoseAnimation> route_animation_;
  gazebo::physics::TrajectoryInfoPtr custom_trajectory_;
  ignition::math::Pose3d controlled_pose_;
  ignition::math::Pose3d actor_pose_offset_{
    0.0, 0.0, 1.2138, 1.5707, 0.0, 1.5707};
  gazebo::common::Time last_update_time_;

  std::mutex mutex_;
  std::string service_prefix_;
  double route_duration_{0.0};
  double route_time_{0.0};
  double animation_time_{0.0};
  double animation_factor_{4.5};
  bool running_{false};
  bool reset_requested_{false};
  bool time_initialized_{false};
};

GZ_REGISTER_MODEL_PLUGIN(WalkingTargetController)

}  // namespace walking_target_controller
