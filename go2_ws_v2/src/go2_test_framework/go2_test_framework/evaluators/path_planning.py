"""State machine for latest-generation T3 endpoint evaluation."""


class PathPlanningEvaluator:
    def __init__(self, endpoint_tolerance_m=1.0, path_timeout_sec=300.0):
        self.endpoint_tolerance_m = float(endpoint_tolerance_m)
        self.path_timeout_sec = float(path_timeout_sec)
        self.latest_generation = None
        self.first_dispatch_time = None
        self.frame_id = None
        self.navigation_dogs = ()
        self.goals = {}
        self.complete = False
        self.path_success = False
        self.failure_reason = None

    def observe_event(self, event):
        generation = event["generation"]
        if event["event"] == "DISPATCHED":
            if self.latest_generation is None or generation > self.latest_generation:
                self.latest_generation = generation
                if self.first_dispatch_time is None:
                    stamp = event["stamp"]
                    self.first_dispatch_time = (
                        float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
                    )
                self.frame_id = event["frame_id"]
                self.navigation_dogs = tuple(event["navigation_dogs"])
                self.goals = event["goals"]
            return
        if generation != self.latest_generation:
            return
        # Dynamic goals are refreshed continuously, so one Nav2 abort does not
        # determine the overall endpoint result.  Keep it as a recorded event
        # and let pose convergence or the path timeout finish the evaluation.
        if event["event"] == "REJECTED":
            self.complete = True
            self.failure_reason = "nav_goal_rejected"

    def observe_errors(self, eval_time, endpoint_errors):
        if self.complete or self.latest_generation is None:
            return
        if set(endpoint_errors) != set(self.navigation_dogs):
            raise ValueError("endpoint errors must match current navigation dogs")
        if all(
            float(endpoint_errors[name]) <= self.endpoint_tolerance_m
            for name in self.navigation_dogs
        ):
            self.complete = True
            self.path_success = True
            return
        if eval_time - self.first_dispatch_time >= self.path_timeout_sec:
            self.complete = True
            self.failure_reason = "path_timeout"
