#include <assimp/Importer.hpp>
#include <assimp/config.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <tinyxml2.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using ignition::math::Pose3d;
using ignition::math::Vector3d;
using tinyxml2::XMLElement;

namespace
{

struct Options
{
  fs::path world;
  fs::path output_prefix;
  double min_x{0.0};
  double min_y{0.0};
  double max_x{0.0};
  double max_y{0.0};
  double resolution{0.2};
  double z_min{0.03};
  double z_max{0.8};
  int border_cells{1};
  std::vector<fs::path> model_paths;
  std::set<std::string> ignored_models;
};

struct Aabb
{
  Vector3d min{
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};
  Vector3d max{
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()};

  void extend(const Vector3d & point)
  {
    min.X(std::min(min.X(), point.X()));
    min.Y(std::min(min.Y(), point.Y()));
    min.Z(std::min(min.Z(), point.Z()));
    max.X(std::max(max.X(), point.X()));
    max.Y(std::max(max.Y(), point.Y()));
    max.Z(std::max(max.Z(), point.Z()));
  }

  bool valid() const
  {
    return std::isfinite(min.X()) && std::isfinite(min.Y()) && std::isfinite(min.Z()) &&
           std::isfinite(max.X()) && std::isfinite(max.Y()) && std::isfinite(max.Z()) &&
           min.X() <= max.X() && min.Y() <= max.Y() && min.Z() <= max.Z();
  }
};

struct CollisionRecord
{
  std::string model;
  std::string link;
  std::string collision;
  std::string geometry;
  bool is_static{false};
  std::string status;
  std::optional<Aabb> aabb;
  std::string source_uri;
};

struct ExpandedSdf
{
  std::string xml;
  std::string diagnostics;
};

std::string usage()
{
  return R"(Usage:
  world_to_grid --world WORLD --output-prefix PREFIX
    --bounds MIN_X MIN_Y MAX_X MAX_Y [options]

Required:
  --world PATH                 Gazebo SDF world file
  --output-prefix PATH         Output prefix for .pgm/.yaml/_collisions.csv/_preview.svg
  --bounds XMIN YMIN XMAX YMAX World-coordinate map bounds

Options:
  --resolution METERS         Cell resolution (default: 0.20)
  --z-min METERS              Lowest blocking height (default: 0.03)
  --z-max METERS              Highest blocking height (default: 0.80)
  --border-cells COUNT        Occupied map border width (default: 1)
  --model-path PATH           Additional model:// search root; repeatable
  --ignore-model NAME         Ignore a model and its descendants; repeatable
  -h, --help                  Show this help
)";
}

double parseDouble(const std::string & text, const std::string & option)
{
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error(option + " requires a number, got: " + text);
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error(option + " requires a finite number, got: " + text);
  }
  return value;
}

int parseInt(const std::string & text, const std::string & option)
{
  std::size_t consumed = 0;
  long value = 0;
  try {
    value = std::stol(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error(option + " requires an integer, got: " + text);
  }
  if (consumed != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(option + " requires a non-negative integer, got: " + text);
  }
  return static_cast<int>(value);
}

Options parseArgs(int argc, char ** argv)
{
  Options options;
  bool have_world = false;
  bool have_output = false;
  bool have_bounds = false;

  auto requireValues = [&](int index, int count, const std::string & option) {
      if (index + count >= argc) {
        throw std::runtime_error(option + " is missing required value(s)");
      }
    };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << usage();
      std::exit(0);
    } else if (arg == "--world") {
      requireValues(i, 1, arg);
      options.world = argv[++i];
      have_world = true;
    } else if (arg == "--output-prefix") {
      requireValues(i, 1, arg);
      options.output_prefix = argv[++i];
      have_output = true;
    } else if (arg == "--bounds") {
      requireValues(i, 4, arg);
      options.min_x = parseDouble(argv[++i], arg);
      options.min_y = parseDouble(argv[++i], arg);
      options.max_x = parseDouble(argv[++i], arg);
      options.max_y = parseDouble(argv[++i], arg);
      have_bounds = true;
    } else if (arg == "--resolution") {
      requireValues(i, 1, arg);
      options.resolution = parseDouble(argv[++i], arg);
    } else if (arg == "--z-min") {
      requireValues(i, 1, arg);
      options.z_min = parseDouble(argv[++i], arg);
    } else if (arg == "--z-max") {
      requireValues(i, 1, arg);
      options.z_max = parseDouble(argv[++i], arg);
    } else if (arg == "--border-cells") {
      requireValues(i, 1, arg);
      options.border_cells = parseInt(argv[++i], arg);
    } else if (arg == "--model-path") {
      requireValues(i, 1, arg);
      options.model_paths.emplace_back(argv[++i]);
    } else if (arg == "--ignore-model") {
      requireValues(i, 1, arg);
      options.ignored_models.emplace(argv[++i]);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (!have_world || !have_output || !have_bounds) {
    throw std::runtime_error("--world, --output-prefix and --bounds are required");
  }
  if (!fs::is_regular_file(options.world)) {
    throw std::runtime_error("world file does not exist: " + options.world.string());
  }
  if (!(options.max_x > options.min_x) || !(options.max_y > options.min_y)) {
    throw std::runtime_error("map bounds must have positive width and height");
  }
  if (!(options.resolution > 0.0)) {
    throw std::runtime_error("--resolution must be greater than zero");
  }
  if (!(options.z_max > options.z_min)) {
    throw std::runtime_error("--z-max must be greater than --z-min");
  }

  const double width_cells = (options.max_x - options.min_x) / options.resolution;
  const double height_cells = (options.max_y - options.min_y) / options.resolution;
  if (std::abs(width_cells - std::round(width_cells)) > 1e-8 ||
    std::abs(height_cells - std::round(height_cells)) > 1e-8)
  {
    throw std::runtime_error("map width and height must be integer multiples of resolution");
  }
  if (options.border_cells * 2 >= std::round(width_cells) ||
    options.border_cells * 2 >= std::round(height_cells))
  {
    throw std::runtime_error("--border-cells is too large for the selected map");
  }
  return options;
}

std::string readFile(const fs::path & path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open temporary file: " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

fs::path makeTempFile(const char * pattern)
{
  std::vector<char> buffer(pattern, pattern + std::strlen(pattern) + 1);
  const int fd = mkstemp(buffer.data());
  if (fd < 0) {
    throw std::runtime_error(std::string("mkstemp failed: ") + std::strerror(errno));
  }
  close(fd);
  return fs::path(buffer.data());
}

std::vector<fs::path> splitModelPaths(const Options & options)
{
  std::vector<fs::path> result = options.model_paths;
  const char * env = std::getenv("GAZEBO_MODEL_PATH");
  if (env != nullptr) {
    std::stringstream stream(env);
    std::string item;
    while (std::getline(stream, item, ':')) {
      if (!item.empty()) {
        result.emplace_back(item);
      }
    }
  }
  const char * home = std::getenv("HOME");
  if (home != nullptr) {
    result.emplace_back(fs::path(home) / ".gazebo" / "models");
  }

  std::vector<fs::path> unique;
  std::set<std::string> seen;
  for (const auto & path : result) {
    const fs::path normalized = fs::absolute(path).lexically_normal();
    if (seen.emplace(normalized.string()).second) {
      unique.push_back(normalized);
    }
  }
  return unique;
}

std::string joinedModelPath(const std::vector<fs::path> & paths)
{
  std::ostringstream stream;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    if (i > 0) {
      stream << ':';
    }
    stream << paths[i].string();
  }
  return stream.str();
}

ExpandedSdf expandSdf(const Options & options, const std::vector<fs::path> & model_paths)
{
  const fs::path stdout_path = makeTempFile("/tmp/world_to_grid_stdout_XXXXXX");
  const fs::path stderr_path = makeTempFile("/tmp/world_to_grid_stderr_XXXXXX");
  struct TempCleanup
  {
    fs::path out;
    fs::path err;
    ~TempCleanup()
    {
      std::error_code ignored;
      fs::remove(out, ignored);
      fs::remove(err, ignored);
    }
  } cleanup{stdout_path, stderr_path};

  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
  }
  if (pid == 0) {
    const int out_fd = open(stdout_path.c_str(), O_WRONLY | O_TRUNC);
    const int err_fd = open(stderr_path.c_str(), O_WRONLY | O_TRUNC);
    if (out_fd < 0 || err_fd < 0 || dup2(out_fd, STDOUT_FILENO) < 0 ||
      dup2(err_fd, STDERR_FILENO) < 0)
    {
      _exit(126);
    }
    close(out_fd);
    close(err_fd);
    const std::string model_path = joinedModelPath(model_paths);
    if (!model_path.empty()) {
      setenv("GAZEBO_MODEL_PATH", model_path.c_str(), 1);
    }
    execlp("gz", "gz", "sdf", "-p", options.world.c_str(), static_cast<char *>(nullptr));
    _exit(errno == ENOENT ? 127 : 126);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    throw std::runtime_error(std::string("waitpid failed: ") + std::strerror(errno));
  }
  const std::string diagnostics = readFile(stderr_path);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ostringstream message;
    message << "gz sdf -p failed";
    if (WIFEXITED(status)) {
      message << " with exit code " << WEXITSTATUS(status);
    }
    if (!diagnostics.empty()) {
      message << ":\n" << diagnostics;
    }
    throw std::runtime_error(message.str());
  }
  const std::string xml = readFile(stdout_path);
  if (xml.empty()) {
    throw std::runtime_error("gz sdf -p produced empty output");
  }
  return {xml, diagnostics};
}

std::vector<double> parseNumbers(
  const char * text, std::size_t count, const std::string & context)
{
  if (text == nullptr) {
    throw std::runtime_error(context + " is missing numeric content");
  }
  std::istringstream stream(text);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(context + " contains a non-finite value");
    }
    values.push_back(value);
  }
  stream >> std::ws;
  if (!stream.eof() || values.size() != count) {
    throw std::runtime_error(
            context + " requires exactly " + std::to_string(count) + " numbers");
  }
  return values;
}

Pose3d parsePose(const XMLElement * parent, const std::string & context)
{
  const XMLElement * pose = parent->FirstChildElement("pose");
  if (pose == nullptr) {
    return Pose3d::Zero;
  }
  const auto values = parseNumbers(pose->GetText(), 6, context + "/pose");
  return Pose3d(values[0], values[1], values[2], values[3], values[4], values[5]);
}

std::string relativeTo(const XMLElement * parent)
{
  const XMLElement * pose = parent->FirstChildElement("pose");
  if (pose == nullptr) {
    return {};
  }
  const char * value = pose->Attribute("relative_to");
  return value == nullptr ? std::string{} : std::string(value);
}

bool parseBool(const XMLElement * parent, const char * child_name, bool default_value)
{
  const XMLElement * child = parent->FirstChildElement(child_name);
  if (child == nullptr || child->GetText() == nullptr) {
    return default_value;
  }
  const std::string text = child->GetText();
  if (text == "1" || text == "true") {
    return true;
  }
  if (text == "0" || text == "false") {
    return false;
  }
  throw std::runtime_error(
          std::string("invalid boolean in <") + child_name + ">: " + text);
}

std::string elementName(const XMLElement * element, const std::string & fallback)
{
  const char * name = element->Attribute("name");
  return name == nullptr ? fallback : std::string(name);
}

Vector3d transformPoint(const Pose3d & pose, const Vector3d & point)
{
  return pose.Pos() + pose.Rot().RotateVector(point);
}

void addBoxCorners(Aabb & aabb, const Pose3d & pose, const Vector3d & half_size)
{
  for (const double x : {-half_size.X(), half_size.X()}) {
    for (const double y : {-half_size.Y(), half_size.Y()}) {
      for (const double z : {-half_size.Z(), half_size.Z()}) {
        aabb.extend(transformPoint(pose, Vector3d(x, y, z)));
      }
    }
  }
}

aiMatrix4x4 multiply(const aiMatrix4x4 & lhs, const aiMatrix4x4 & rhs)
{
  return lhs * rhs;
}

Vector3d assimpPoint(const aiMatrix4x4 & matrix, const aiVector3D & vertex)
{
  const aiVector3D transformed = matrix * vertex;
  return Vector3d(transformed.x, transformed.y, transformed.z);
}

void collectAssimpVertices(
  const aiScene * scene, const aiNode * node, const aiMatrix4x4 & parent,
  std::vector<Vector3d> & vertices)
{
  const aiMatrix4x4 accumulated = multiply(parent, node->mTransformation);
  for (unsigned int mesh_index = 0; mesh_index < node->mNumMeshes; ++mesh_index) {
    const unsigned int scene_index = node->mMeshes[mesh_index];
    if (scene_index >= scene->mNumMeshes) {
      throw std::runtime_error("Assimp node references an invalid mesh index");
    }
    const aiMesh * mesh = scene->mMeshes[scene_index];
    for (unsigned int vertex_index = 0; vertex_index < mesh->mNumVertices; ++vertex_index) {
      vertices.push_back(assimpPoint(accumulated, mesh->mVertices[vertex_index]));
    }
  }
  for (unsigned int child_index = 0; child_index < node->mNumChildren; ++child_index) {
    collectAssimpVertices(scene, node->mChildren[child_index], accumulated, vertices);
  }
}

class MeshCache
{
public:
  const std::vector<Vector3d> & load(const fs::path & path)
  {
    const fs::path canonical = fs::weakly_canonical(path);
    const std::string key = canonical.string();
    auto found = vertices_.find(key);
    if (found != vertices_.end()) {
      return found->second;
    }

    Assimp::Importer importer;
    // Gazebo/SDF is Z-up.  Assimp otherwise rotates COLLADA Z_UP assets to its
    // internal Y-up convention, which would turn horizontal spans into height.
    // Keep the source up axis while still applying the COLLADA unit transform.
    importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);
    const aiScene * scene = importer.ReadFile(
      key, aiProcess_JoinIdenticalVertices | aiProcess_ValidateDataStructure);
    if (scene == nullptr || scene->mRootNode == nullptr) {
      throw std::runtime_error(
              "Assimp failed to load mesh " + key + ": " + importer.GetErrorString());
    }
    std::vector<Vector3d> vertices;
    collectAssimpVertices(scene, scene->mRootNode, aiMatrix4x4(), vertices);
    if (vertices.empty()) {
      throw std::runtime_error("mesh contains no vertices: " + key);
    }
    return vertices_.emplace(key, std::move(vertices)).first->second;
  }

private:
  std::unordered_map<std::string, std::vector<Vector3d>> vertices_;
};

class Converter
{
public:
  Converter(Options options, std::vector<fs::path> model_paths)
  : options_(std::move(options)), model_paths_(std::move(model_paths))
  {
    width_ = static_cast<int>(std::llround(
        (options_.max_x - options_.min_x) / options_.resolution));
    height_ = static_cast<int>(std::llround(
        (options_.max_y - options_.min_y) / options_.resolution));
    grid_.assign(static_cast<std::size_t>(width_) * height_, 254);
  }

  void parse(const std::string & xml)
  {
    tinyxml2::XMLDocument document;
    const tinyxml2::XMLError error = document.Parse(xml.c_str(), xml.size());
    if (error != tinyxml2::XML_SUCCESS) {
      throw std::runtime_error(
              std::string("failed to parse expanded SDF: ") + document.ErrorStr());
    }
    const XMLElement * sdf = document.FirstChildElement("sdf");
    if (sdf == nullptr) {
      throw std::runtime_error("expanded file has no <sdf> root");
    }
    const XMLElement * world = sdf->FirstChildElement("world");
    if (world == nullptr) {
      throw std::runtime_error("expanded file has no <world>");
    }
    if (world->NextSiblingElement("world") != nullptr) {
      throw std::runtime_error("multiple worlds are not supported by this CLI");
    }
    for (const XMLElement * model = world->FirstChildElement("model"); model != nullptr;
      model = model->NextSiblingElement("model"))
    {
      processModel(model, Pose3d::Zero, Pose3d::Zero, "world");
    }
    applyBorder();
  }

  void writeOutputs() const
  {
    const fs::path parent = options_.output_prefix.parent_path();
    if (!parent.empty()) {
      fs::create_directories(parent);
    }
    const std::string prefix = options_.output_prefix.string();
    const std::array<fs::path, 4> final_paths = {
      fs::path(prefix + ".pgm"),
      fs::path(prefix + ".yaml"),
      fs::path(prefix + "_collisions.csv"),
      fs::path(prefix + "_preview.svg")};
    const std::string suffix = ".tmp." + std::to_string(getpid());
    const std::string backup_suffix = ".backup." + std::to_string(getpid());
    std::array<fs::path, 4> temp_paths;
    std::array<fs::path, 4> backup_paths;
    std::array<bool, 4> had_original{};
    std::array<bool, 4> installed{};
    for (std::size_t i = 0; i < final_paths.size(); ++i) {
      temp_paths[i] = fs::path(final_paths[i].string() + suffix);
      backup_paths[i] = fs::path(final_paths[i].string() + backup_suffix);
    }

    try {
      writePgm(temp_paths[0]);
      writeYaml(temp_paths[1], final_paths[0].filename().string());
      writeCsv(temp_paths[2]);
      writeSvg(temp_paths[3]);
      for (std::size_t i = 0; i < final_paths.size(); ++i) {
        if (fs::exists(final_paths[i])) {
          fs::rename(final_paths[i], backup_paths[i]);
          had_original[i] = true;
        }
      }
      for (std::size_t i = 0; i < final_paths.size(); ++i) {
        fs::rename(temp_paths[i], final_paths[i]);
        installed[i] = true;
      }
      for (std::size_t i = 0; i < final_paths.size(); ++i) {
        if (had_original[i]) {
          std::error_code ignored;
          fs::remove(backup_paths[i], ignored);
        }
      }
    } catch (...) {
      std::error_code ignored;
      for (std::size_t i = 0; i < final_paths.size(); ++i) {
        fs::remove(temp_paths[i], ignored);
        if (installed[i]) {
          fs::remove(final_paths[i], ignored);
        }
      }
      for (std::size_t i = 0; i < final_paths.size(); ++i) {
        if (had_original[i] && fs::exists(backup_paths[i])) {
          fs::rename(backup_paths[i], final_paths[i], ignored);
        }
      }
      throw;
    }
  }

  void printStats() const
  {
    std::map<std::string, std::size_t> statuses;
    for (const auto & record : records_) {
      ++statuses[record.status];
    }
    const std::size_t occupied = static_cast<std::size_t>(
      std::count(grid_.begin(), grid_.end(), static_cast<unsigned char>(0)));
    const double ratio = 100.0 * occupied / static_cast<double>(grid_.size());
    std::cout << "world_to_grid complete\n"
              << "  map: " << width_ << " x " << height_ << " @ "
              << options_.resolution << " m/cell\n"
              << "  collisions/records: " << records_.size() << "\n";
    for (const auto & [status, count] : statuses) {
      std::cout << "  " << status << ": " << count << "\n";
    }
    std::cout << "  occupied cells: " << occupied << " (" << std::fixed
              << std::setprecision(3) << ratio << "%)\n"
              << "  output prefix: " << options_.output_prefix << '\n';
  }

private:
  void recordIgnoredModel(const XMLElement * model)
  {
    const std::size_t first_record = records_.size();
    const std::string model_name = elementName(model, "unnamed_model");
    const bool is_static = parseBool(model, "static", false);
    for (const XMLElement * link = model->FirstChildElement("link"); link != nullptr;
      link = link->NextSiblingElement("link"))
    {
      const std::string link_name = elementName(link, "unnamed_link");
      for (const XMLElement * collision = link->FirstChildElement("collision"); collision != nullptr;
        collision = collision->NextSiblingElement("collision"))
      {
        std::string type = "unknown";
        std::string source_uri;
        if (const XMLElement * geometry = collision->FirstChildElement("geometry")) {
          if (geometry->FirstChildElement("box")) {
            type = "box";
          } else if (geometry->FirstChildElement("cylinder")) {
            type = "cylinder";
          } else if (geometry->FirstChildElement("sphere")) {
            type = "sphere";
          } else if (const XMLElement * mesh = geometry->FirstChildElement("mesh")) {
            type = "mesh";
            if (const XMLElement * uri = mesh->FirstChildElement("uri");
              uri != nullptr && uri->GetText() != nullptr)
            {
              source_uri = uri->GetText();
            }
          } else if (geometry->FirstChildElement("plane")) {
            type = "plane";
          } else if (geometry->FirstChildElement("heightmap")) {
            type = "heightmap";
          }
        }
        records_.push_back(
          {model_name, link_name, elementName(collision, "unnamed_collision"), type,
            is_static, "ignored_model", std::nullopt, source_uri});
      }
    }
    for (const XMLElement * nested = model->FirstChildElement("model"); nested != nullptr;
      nested = nested->NextSiblingElement("model"))
    {
      recordIgnoredModel(nested);
    }
    if (records_.size() == first_record) {
      records_.push_back(
        {model_name, "", "", "model", is_static, "ignored_model", std::nullopt, ""});
    }
  }

  void validateRelativeTo(
    const XMLElement * element, const std::string & context,
    bool allow_model_frame) const
  {
    const std::string relative_to = relativeTo(element);
    if (relative_to.empty()) {
      return;
    }
    if (allow_model_frame && relative_to == "__model__") {
      return;
    }
    throw std::runtime_error(
            context + " uses unsupported pose relative_to='" + relative_to + "'");
  }

  void processModel(
    const XMLElement * model, const Pose3d & parent_world,
    const Pose3d & parent_model_world, const std::string & parent_context)
  {
    const std::string name = elementName(model, "unnamed_model");
    const std::string context = parent_context + "/model[" + name + "]";
    if (options_.ignored_models.count(name) > 0) {
      recordIgnoredModel(model);
      return;
    }

    validateRelativeTo(model, context, true);
    const Pose3d model_local = parsePose(model, context);
    const Pose3d base = relativeTo(model) == "__model__" ? parent_model_world : parent_world;
    const Pose3d model_world = base * model_local;
    const bool is_static = parseBool(model, "static", false);

    for (const XMLElement * link = model->FirstChildElement("link"); link != nullptr;
      link = link->NextSiblingElement("link"))
    {
      processLink(link, model_world, name, is_static, context);
    }
    for (const XMLElement * nested = model->FirstChildElement("model"); nested != nullptr;
      nested = nested->NextSiblingElement("model"))
    {
      processModel(nested, model_world, model_world, context);
    }
  }

  void processLink(
    const XMLElement * link, const Pose3d & model_world, const std::string & model_name,
    bool is_static, const std::string & model_context)
  {
    const std::string link_name = elementName(link, "unnamed_link");
    const std::string context = model_context + "/link[" + link_name + "]";
    validateRelativeTo(link, context, true);
    const Pose3d link_local = parsePose(link, context);
    const Pose3d link_world = model_world * link_local;

    for (const XMLElement * collision = link->FirstChildElement("collision"); collision != nullptr;
      collision = collision->NextSiblingElement("collision"))
    {
      processCollision(
        collision, model_world, link_world, model_name, link_name, is_static, context);
    }
  }

  fs::path resolveMeshUri(const std::string & uri) const
  {
    if (uri.rfind("model://", 0) == 0) {
      const fs::path relative = uri.substr(std::strlen("model://"));
      for (const auto & root : model_paths_) {
        const fs::path candidate = root / relative;
        if (fs::is_regular_file(candidate)) {
          return candidate;
        }
      }
      throw std::runtime_error("unable to resolve mesh URI: " + uri);
    }
    if (uri.rfind("file://", 0) == 0) {
      fs::path path = uri.substr(std::strlen("file://"));
      if (path.is_relative()) {
        path = options_.world.parent_path() / path;
      }
      if (!fs::is_regular_file(path)) {
        throw std::runtime_error("mesh file does not exist: " + path.string());
      }
      return path;
    }
    fs::path path(uri);
    if (path.is_relative()) {
      path = options_.world.parent_path() / path;
    }
    if (!fs::is_regular_file(path)) {
      throw std::runtime_error("mesh file does not exist: " + path.string());
    }
    return path;
  }

  Aabb geometryAabb(
    const XMLElement * geometry, const Pose3d & collision_world,
    std::string & type, std::string & source_uri, const std::string & context)
  {
    Aabb aabb;
    if (const XMLElement * box = geometry->FirstChildElement("box")) {
      type = "box";
      const XMLElement * size_element = box->FirstChildElement("size");
      const auto values = parseNumbers(
        size_element == nullptr ? nullptr : size_element->GetText(), 3, context + "/box/size");
      const Vector3d size(values[0], values[1], values[2]);
      if (size.X() <= 0.0 || size.Y() <= 0.0 || size.Z() <= 0.0) {
        throw std::runtime_error(context + " has non-positive box size");
      }
      addBoxCorners(aabb, collision_world, size * 0.5);
    } else if (const XMLElement * cylinder = geometry->FirstChildElement("cylinder")) {
      type = "cylinder";
      const XMLElement * radius_element = cylinder->FirstChildElement("radius");
      const XMLElement * length_element = cylinder->FirstChildElement("length");
      const double radius = parseNumbers(
        radius_element == nullptr ? nullptr : radius_element->GetText(), 1,
        context + "/cylinder/radius")[0];
      const double length = parseNumbers(
        length_element == nullptr ? nullptr : length_element->GetText(), 1,
        context + "/cylinder/length")[0];
      if (radius <= 0.0 || length <= 0.0) {
        throw std::runtime_error(context + " has non-positive cylinder dimensions");
      }
      addBoxCorners(aabb, collision_world, Vector3d(radius, radius, length * 0.5));
    } else if (const XMLElement * sphere = geometry->FirstChildElement("sphere")) {
      type = "sphere";
      const XMLElement * radius_element = sphere->FirstChildElement("radius");
      const double radius = parseNumbers(
        radius_element == nullptr ? nullptr : radius_element->GetText(), 1,
        context + "/sphere/radius")[0];
      if (radius <= 0.0) {
        throw std::runtime_error(context + " has non-positive sphere radius");
      }
      addBoxCorners(aabb, collision_world, Vector3d(radius, radius, radius));
    } else if (const XMLElement * mesh = geometry->FirstChildElement("mesh")) {
      type = "mesh";
      if (mesh->FirstChildElement("submesh") != nullptr) {
        throw std::runtime_error(context + " uses unsupported mesh submesh");
      }
      const XMLElement * uri_element = mesh->FirstChildElement("uri");
      if (uri_element == nullptr || uri_element->GetText() == nullptr ||
        std::string(uri_element->GetText()).empty())
      {
        throw std::runtime_error(context + " mesh has no URI");
      }
      source_uri = uri_element->GetText();
      Vector3d scale(1.0, 1.0, 1.0);
      if (const XMLElement * scale_element = mesh->FirstChildElement("scale")) {
        const auto values = parseNumbers(scale_element->GetText(), 3, context + "/mesh/scale");
        scale.Set(values[0], values[1], values[2]);
      }
      if (scale.X() <= 0.0 || scale.Y() <= 0.0 || scale.Z() <= 0.0) {
        throw std::runtime_error(context + " has non-positive mesh scale");
      }
      const fs::path path = resolveMeshUri(source_uri);
      const auto & vertices = mesh_cache_.load(path);
      for (const auto & vertex : vertices) {
        const Vector3d scaled(
          vertex.X() * scale.X(), vertex.Y() * scale.Y(), vertex.Z() * scale.Z());
        aabb.extend(transformPoint(collision_world, scaled));
      }
    } else if (geometry->FirstChildElement("plane") != nullptr) {
      throw std::runtime_error(context + " uses unsupported plane geometry");
    } else if (geometry->FirstChildElement("heightmap") != nullptr) {
      throw std::runtime_error(context + " uses unsupported heightmap geometry");
    } else {
      throw std::runtime_error(context + " has unsupported or empty geometry");
    }
    if (!aabb.valid()) {
      throw std::runtime_error(context + " produced an invalid AABB");
    }
    return aabb;
  }

  void processCollision(
    const XMLElement * collision, const Pose3d & model_world, const Pose3d & link_world,
    const std::string & model_name, const std::string & link_name, bool is_static,
    const std::string & link_context)
  {
    const std::string collision_name = elementName(collision, "unnamed_collision");
    const std::string context = link_context + "/collision[" + collision_name + "]";
    validateRelativeTo(collision, context, true);
    const Pose3d collision_local = parsePose(collision, context);
    const Pose3d collision_world =
      relativeTo(collision) == "__model__" ? model_world * collision_local :
      link_world * collision_local;
    const XMLElement * geometry = collision->FirstChildElement("geometry");
    if (geometry == nullptr) {
      throw std::runtime_error(context + " has no <geometry>");
    }

    std::string type;
    std::string source_uri;
    const Aabb aabb = geometryAabb(geometry, collision_world, type, source_uri, context);
    std::string status;
    if (aabb.max.Z() < options_.z_min) {
      status = "ignored_below_slice";
    } else if (aabb.min.Z() > options_.z_max) {
      status = "ignored_above_slice";
    } else if (aabb.max.X() <= options_.min_x || aabb.min.X() >= options_.max_x ||
      aabb.max.Y() <= options_.min_y || aabb.min.Y() >= options_.max_y)
    {
      status = "ignored_outside_bounds";
    } else {
      status = "occupied";
      rasterize(aabb);
    }
    records_.push_back(
      {model_name, link_name, collision_name, type, is_static, status, aabb, source_uri});
  }

  void rasterize(const Aabb & aabb)
  {
    int min_gx = static_cast<int>(std::floor(
        (aabb.min.X() - options_.min_x) / options_.resolution));
    int max_gx = static_cast<int>(std::ceil(
        (aabb.max.X() - options_.min_x) / options_.resolution)) - 1;
    int min_gy = static_cast<int>(std::floor(
        (aabb.min.Y() - options_.min_y) / options_.resolution));
    int max_gy = static_cast<int>(std::ceil(
        (aabb.max.Y() - options_.min_y) / options_.resolution)) - 1;
    min_gx = std::clamp(min_gx, 0, width_ - 1);
    max_gx = std::clamp(max_gx, 0, width_ - 1);
    min_gy = std::clamp(min_gy, 0, height_ - 1);
    max_gy = std::clamp(max_gy, 0, height_ - 1);
    for (int gy = min_gy; gy <= max_gy; ++gy) {
      const int row = height_ - 1 - gy;
      for (int gx = min_gx; gx <= max_gx; ++gx) {
        grid_[static_cast<std::size_t>(row) * width_ + gx] = 0;
      }
    }
  }

  void applyBorder()
  {
    for (int layer = 0; layer < options_.border_cells; ++layer) {
      const int left = layer;
      const int right = width_ - 1 - layer;
      const int top = layer;
      const int bottom = height_ - 1 - layer;
      for (int x = left; x <= right; ++x) {
        grid_[static_cast<std::size_t>(top) * width_ + x] = 0;
        grid_[static_cast<std::size_t>(bottom) * width_ + x] = 0;
      }
      for (int y = top; y <= bottom; ++y) {
        grid_[static_cast<std::size_t>(y) * width_ + left] = 0;
        grid_[static_cast<std::size_t>(y) * width_ + right] = 0;
      }
    }
  }

  static std::string csvEscape(const std::string & value)
  {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
      return value;
    }
    std::string escaped = "\"";
    for (const char ch : value) {
      escaped += ch == '\"' ? "\"\"" : std::string(1, ch);
    }
    return escaped + "\"";
  }

  static std::string xmlEscape(const std::string & value)
  {
    std::string result;
    for (const char ch : value) {
      switch (ch) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '\"': result += "&quot;"; break;
        case '\'': result += "&apos;"; break;
        default: result += ch; break;
      }
    }
    return result;
  }

  void writePgm(const fs::path & path) const
  {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw std::runtime_error("failed to create PGM: " + path.string());
    }
    stream << "P5\n# generated by world_to_grid from "
           << options_.world.filename().string() << "\n"
           << width_ << ' ' << height_ << "\n255\n";
    stream.write(reinterpret_cast<const char *>(grid_.data()), grid_.size());
    if (!stream) {
      throw std::runtime_error("failed while writing PGM: " + path.string());
    }
  }

  void writeYaml(const fs::path & path, const std::string & image_name) const
  {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
      throw std::runtime_error("failed to create YAML: " + path.string());
    }
    stream << std::setprecision(12)
           << "image: " << image_name << '\n'
           << "mode: trinary\n"
           << "resolution: " << options_.resolution << '\n'
           << "origin: [" << options_.min_x << ", " << options_.min_y << ", 0.0]\n"
           << "negate: 0\n"
           << "occupied_thresh: 0.65\n"
           << "free_thresh: 0.25\n";
  }

  void writeCsv(const fs::path & path) const
  {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
      throw std::runtime_error("failed to create CSV: " + path.string());
    }
    stream << "model,link,collision,geometry,static,status,"
              "min_x,min_y,min_z,max_x,max_y,max_z,source_uri\n";
    stream << std::setprecision(12);
    for (const auto & record : records_) {
      stream << csvEscape(record.model) << ',' << csvEscape(record.link) << ','
             << csvEscape(record.collision) << ',' << csvEscape(record.geometry) << ','
             << (record.is_static ? "true" : "false") << ',' << record.status << ',';
      if (record.aabb.has_value()) {
        const Aabb & aabb = *record.aabb;
        stream << aabb.min.X() << ',' << aabb.min.Y() << ',' << aabb.min.Z() << ','
               << aabb.max.X() << ',' << aabb.max.Y() << ',' << aabb.max.Z();
      } else {
        stream << ",,,,,";
      }
      stream << ',' << csvEscape(record.source_uri) << '\n';
    }
  }

  void writeSvg(const fs::path & path) const
  {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
      throw std::runtime_error("failed to create SVG: " + path.string());
    }
    stream << std::setprecision(12)
           << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
           << width_ << ' ' << height_ << "\">\n"
           << "<rect width=\"" << width_ << "\" height=\"" << height_
           << "\" fill=\"white\" stroke=\"black\"/>\n";
    for (const auto & record : records_) {
      if (!record.aabb.has_value() || record.status == "ignored_outside_bounds") {
        continue;
      }
      const Aabb & aabb = *record.aabb;
      const double left = (aabb.min.X() - options_.min_x) / options_.resolution;
      const double right = (aabb.max.X() - options_.min_x) / options_.resolution;
      const double top = height_ - (aabb.max.Y() - options_.min_y) / options_.resolution;
      const double bottom = height_ - (aabb.min.Y() - options_.min_y) / options_.resolution;
      const double x = std::clamp(left, 0.0, static_cast<double>(width_));
      const double y = std::clamp(top, 0.0, static_cast<double>(height_));
      const double rect_width = std::max(
        0.0, std::min(right, static_cast<double>(width_)) - x);
      const double rect_height = std::max(
        0.0, std::min(bottom, static_cast<double>(height_)) - y);
      if (rect_width <= 0.0 || rect_height <= 0.0) {
        continue;
      }
      std::string style;
      if (record.status == "occupied") {
        style = "fill:black;fill-opacity:0.65;stroke:black;stroke-width:0.4";
      } else if (record.status == "ignored_below_slice") {
        style = "fill:none;stroke:#20a020;stroke-width:0.6";
      } else {
        style = "fill:none;stroke:#888;stroke-width:0.6";
      }
      stream << "<g><title>" << xmlEscape(record.model + "/" + record.collision +
        " [" + record.status + "]") << "</title>"
             << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\""
             << rect_width << "\" height=\"" << rect_height << "\" style=\""
             << style << "\"/>";
      const char * label_color = record.status == "occupied" ? "#b00000" :
        (record.status == "ignored_below_slice" ? "#087808" : "#666");
      stream << "<text x=\"" << x << "\" y=\"" << std::max(5.0, y)
             << "\" font-size=\"4\" fill=\"" << label_color << "\">"
             << xmlEscape(record.model) << "</text>";
      stream << "</g>\n";
    }
    stream << "</svg>\n";
  }

  Options options_;
  std::vector<fs::path> model_paths_;
  int width_{0};
  int height_{0};
  std::vector<unsigned char> grid_;
  std::vector<CollisionRecord> records_;
  MeshCache mesh_cache_;
};

}  // namespace

int main(int argc, char ** argv)
{
  try {
    Options options = parseArgs(argc, argv);
    const std::vector<fs::path> model_paths = splitModelPaths(options);
    const ExpandedSdf expanded = expandSdf(options, model_paths);
    if (!expanded.diagnostics.empty()) {
      std::cerr << expanded.diagnostics;
    }
    Converter converter(std::move(options), model_paths);
    converter.parse(expanded.xml);
    converter.writeOutputs();
    converter.printStats();
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "world_to_grid error: " << error.what() << '\n';
    return 1;
  }
}
