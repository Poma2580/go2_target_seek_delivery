"""Pure Nav2-to-MADDPG handoff state machine."""

from enum import Enum


class HandoffState(str, Enum):
    ROLE_WAIT = "ROLE_WAIT"
    NAV2_ACTIVE = "NAV2_ACTIVE"
    ARRIVAL_HOLD = "ARRIVAL_HOLD"
    NAV2_CANCELLING = "NAV2_CANCELLING"
    WAITING_FOR_STOP = "WAITING_FOR_STOP"
    WAITING_FOR_MADDPG_READY = "WAITING_FOR_MADDPG_READY"
    ENABLING_MADDPG = "ENABLING_MADDPG"
    MADDPG_ACTIVE = "MADDPG_ACTIVE"
    HANDOFF_FAILED = "HANDOFF_FAILED"


class HandoffManager:
    """Advance a fail-closed handoff and emit one-shot transition actions."""

    def __init__(self, config, on_cancel, on_enable, on_mux, on_state):
        self.config = config
        self.on_cancel = on_cancel
        self.on_enable = on_enable
        self.on_mux = on_mux
        self.on_state = on_state
        self.state = HandoffState.ROLE_WAIT
        self.state_since = None
        self.hold_since = None
        self.failure_reason = None
        self.on_enable(False)
        self.on_mux(False)
        self.on_state(self.state.value)

    def select_role(self, now):
        if self.state == HandoffState.ROLE_WAIT:
            self._transition(HandoffState.NAV2_ACTIVE, now)

    def update(
        self,
        now,
        arrived=False,
        stopped=False,
        cancel_complete=False,
        cancel_failed=False,
        maddpg_ready=False,
        maddpg_active=False,
    ):
        if self.state in (HandoffState.ROLE_WAIT, HandoffState.HANDOFF_FAILED):
            return

        elapsed = now - self.state_since
        if self.state == HandoffState.NAV2_ACTIVE:
            if arrived:
                self.hold_since = now
                self._transition(HandoffState.ARRIVAL_HOLD, now)
        elif self.state == HandoffState.ARRIVAL_HOLD:
            if not arrived:
                self.hold_since = None
                self._transition(HandoffState.NAV2_ACTIVE, now)
            elif now - self.hold_since >= self.config.arrival_hold_duration:
                self.on_cancel()
                self._transition(HandoffState.NAV2_CANCELLING, now)
        elif self.state == HandoffState.NAV2_CANCELLING:
            if cancel_failed:
                self.fail("Nav2 cancellation failed", now)
            elif cancel_complete:
                self.hold_since = None
                self._transition(HandoffState.WAITING_FOR_STOP, now)
            elif elapsed > self.config.cancel_timeout:
                self.fail("timed out waiting for Nav2 cancellation", now)
        elif self.state == HandoffState.WAITING_FOR_STOP:
            if stopped:
                if self.hold_since is None:
                    self.hold_since = now
                elif now - self.hold_since >= self.config.stopped_hold_duration:
                    self._transition(HandoffState.WAITING_FOR_MADDPG_READY, now)
            else:
                self.hold_since = None
            if elapsed > self.config.stop_timeout:
                self.fail("timed out waiting for navigation dogs to stop", now)
        elif self.state == HandoffState.WAITING_FOR_MADDPG_READY:
            if maddpg_ready:
                self.on_enable(True)
                self._transition(HandoffState.ENABLING_MADDPG, now)
            elif elapsed > self.config.maddpg_ready_timeout:
                self.fail("timed out waiting for MADDPG ready", now)
        elif self.state == HandoffState.ENABLING_MADDPG:
            if maddpg_active:
                # Continuous-control MADDPG owns cmd_vel after handoff.  The
                # waypoint policy only selects goals, so Nav2 must remain the
                # velocity owner in that mode.
                self.on_mux(self.config.switch_mux_to_maddpg)
                self._transition(HandoffState.MADDPG_ACTIVE, now)
            elif elapsed > self.config.maddpg_enable_timeout:
                self.fail("timed out waiting for MADDPG active", now)
        elif self.state == HandoffState.MADDPG_ACTIVE and not maddpg_active:
            self.fail("MADDPG active status dropped", now)

    def fail(self, reason, now):
        if self.state == HandoffState.HANDOFF_FAILED:
            return
        self.failure_reason = reason
        self.on_enable(False)
        self.on_mux(False)
        self._transition(HandoffState.HANDOFF_FAILED, now)

    def shutdown(self):
        self.on_enable(False)
        self.on_mux(False)

    def _transition(self, state, now):
        self.state = state
        self.state_since = now
        self.on_state(state.value)
