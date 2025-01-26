class ScenarioManager:
    def __init__(self, timeout, car_length=2.4508, npc_initial_x):
        """
        :param timeout: Scenario timeout value
        :param car_length: Length of the car (default: 2.4508 meters)
        :param npc_initial_x: Initial X coordinate of the NPC
        """
        self._timeout = timeout
        self.distance = 0
        self.overTTC = []
        self.t1 = []
        self.index = 0
        self.dac = []
        self.TET = 0
        self.TIT = 0
        self.average_dacc = 0
        self.car_length = car_length
        self.npc_initial_x = npc_initial_x  # Replace hardcoded NPC X-coordinate

    def cal_speed(self, actor):
        """Calculate the speed of an actor."""
        return math.sqrt(actor.get_velocity().x**2 + actor.get_velocity().y**2)

    def cal_rela_loc(self, actor, pes):
        """Calculate the relative distance between two actors."""
        dx = actor.get_location().x - pes.get_location().x
        dy = actor.get_location().y - pes.get_location().y
        return math.sqrt(dx**2 + dy**2)

    def cal_rela_speed(self, actor, pes):
        """
        Calculate the relative speed between two actors.
        """
        rela_loc = self.cal_rela_loc(actor, pes)
        current_dis = actor.get_location().x - self.npc_initial_x
        cos_rate = current_dis / rela_loc if rela_loc > 0 else 0
        v_a = self.cal_speed(actor) * cos_rate
        v_p = self.cal_speed(pes) * math.sqrt(1 - cos_rate**2) if cos_rate**2 < 1 else 0
        return v_a + v_p

    def call_TTC(self, actor, pes):
        """Calculate Time-to-Collision (TTC)."""
        loc = self.cal_rela_loc(actor, pes)
        velocity = self.cal_rela_speed(actor, pes)
        if velocity > 0:
            return round((loc - self.car_length) / velocity, 3)
        else:
            return float('inf')  # Avoid division by zero

    def call_TET(self):
        """Calculate Time Exposed Time-to-Collision (TET)."""
        return round(len(self.overTTC) * 0.05, 3)

    def call_TIT(self):
        """Calculate Time Integrated Time-to-Collision (TIT)."""
        return round(sum(1.5 - t for t in self.overTTC if t <= 1.5) * 0.05, 3)

    def _tick_scenario(self, timestamp, ego_vehicle, npc_actor):
        """
        Perform a tick of the scenario to update metrics.
        """
        ego_loc_x = ego_vehicle.get_transform().location.x
        ego_velocity_x = ego_vehicle.get_velocity().x

        # Collect TTC values if ego vehicle passes the NPC
        if ego_loc_x >= self.npc_initial_x + self.car_length:
            ttc = self.call_TTC(ego_vehicle, npc_actor)
            if abs(ttc) <= 1.5:
                self.t1.append(ttc)

        # If the ego vehicle stops, finalize metrics collection
        if abs(ego_velocity_x) < 0.001:
            self.distance = ego_loc_x - self.npc_initial_x - self.car_length
            if ego_loc_x < self.npc_initial_x + 10 and self.index == 0:
                self.overTTC.extend(self.t1)
                self.index = 1

    def stop_scenario(self):
        """
        Stop the scenario and finalize metrics calculation.
        """
        self.TET = self.call_TET()
        self.TIT = self.call_TIT()
        if self.dac:
            self.average_dacc = round(sum(abs(d) for d in self.dac) / len(self.dac), 3)
        self._reset_metrics()

    def _reset_metrics(self):
        """
        Reset all metric-related variables.
        """
        self.distance = 0
        self.overTTC.clear()
        self.t1.clear()
        self.index = 0
        self.dac.clear()