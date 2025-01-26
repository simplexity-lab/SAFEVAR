class NoCrashEvalScenario(RouteScenario):
    """
    A scenario for evaluating autonomous agents in CARLA with various configurations.
    """

    category = "NoCrashEvalScenario"

    def __init__(self, world, agent, start_idx, target_idx, weather_idx, traffic_idx, debug_mode=0, physics=[], criteria_enable=True):
        self.town_name = world.get_map().name
        self.weather_idx = weather_idx
        self.start_idx = start_idx
        self.target_idx = target_idx
        self.traffic_idx = traffic_idx
        self.agent = agent
        self.physics = physics

        # Set route and initialize NPCs
        self._set_route()
        self.npc_id = None  # For identifying the NPC actor
        ego_vehicle = self._update_ego_vehicle(self.physics)

        # Initialize the scenario
        BasicScenario.__init__(
            self,
            name=f"NoCrash_{self.town_name}_{traffic_idx}_w{weather_idx}_s{start_idx}_t{target_idx}",
            ego_vehicles=[ego_vehicle],
            config=None,
            world=world,
            debug_mode=debug_mode > 1,
            terminate_on_failure=False,
            criteria_enable=criteria_enable,
        )

    def _set_route(self, hop_resolution=1.0):
        """
        Sets the route for the ego vehicle from the start to the target.
        """
        world = CarlaDataProvider.get_world()
        dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        spawn_points = CarlaDataProvider._spawn_points
        start = spawn_points[self.start_idx]
        target = spawn_points[self.target_idx]

        # Generate the route
        route = grp.trace_route(start.location, target.location)
        self.route = [(w.transform, c) for w, c in route]

        # Update ego vehicle's route
        CarlaDataProvider.set_ego_vehicle_route([(w.transform.location, c) for w, c in route])
        gps_route = location_route_to_gps(self.route, *_get_latlon_ref(world))
        self.agent.set_global_plan(gps_route, self.route)

        # Estimate the timeout for the scenario
        self.timeout = self._estimate_route_timeout()

    def _initialize_actors(self, config):
        """
        Spawns traffic (vehicles and pedestrians) and a special NPC.
        """
        car_amounts = {"Town01": [0, 20, 100], "Town02": [0, 15, 70]}
        ped_amounts = {"Town01": [0, 50, 200], "Town02": [0, 50, 150]}

        car_amount = car_amounts[self.town_name][self.traffic_idx]
        ped_amount = ped_amounts[self.town_name][self.traffic_idx]

        # Spawn vehicles and pedestrians
        CarlaDataProvider.request_new_batch_actors(
            "vehicle.*", car_amount, carla.Transform(), autopilot=True, random_location=True, rolename="background"
        )
        pedestrians = self._spawn_pedestrians(ped_amount)

        # Spawn a special NPC in Town01 under specific conditions
        if self.town_name == "Town01" and car_amount == 0 and ped_amount == 0:
            npc_spawn_point = Transform(Location(x=210.67, y=190.15, z=0.3))
            self.npc_id = self._spawn_npc(npc_spawn_point)

    def _spawn_pedestrians(self, ped_amount):
        """
        Helper method to spawn pedestrians.
        """
        blueprints = CarlaDataProvider._blueprint_library.filter("walker.pedestrian.*")
        spawn_points = []

        while len(spawn_points) < ped_amount:
            loc = CarlaDataProvider.get_world().get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(location=loc))

        batch = [carla.command.SpawnActor(random.choice(blueprints), sp) for sp in spawn_points]
        pedestrians = CarlaDataProvider.handle_actor_batch(batch)
        return pedestrians

    def _spawn_npc(self, spawn_point):
        """
        Helper method to spawn a special NPC pedestrian.
        """
        walker_bp = random.choice(CarlaDataProvider._blueprint_library.filter("walker.pedestrian.0016"))
        batch = [carla.command.SpawnActor(walker_bp, spawn_point)]
        npc = CarlaDataProvider.handle_actor_batch(batch)
        return npc[0].id if npc else None

    def _initialize_environment(self, world):
        """
        Set the weather for the scenario.
        """
        world.set_weather(WEATHERS[self.weather_idx])

    def _create_test_criteria(self):
        """
        Define evaluation criteria for the scenario.
        """
        criteria = []
        route = convert_transform_to_location(self.route)

        criteria.append(RouteCompletionTest(self.ego_vehicles[0], route=route))
        criteria.append(OutsideRouteLanesTest(self.ego_vehicles[0], route=route))
        criteria.append(CollisionTest(self.ego_vehicles[0]))
        criteria.append(RunningRedLightTest(self.ego_vehicles[0]))
        criteria.append(RunningStopTest(self.ego_vehicles[0]))
        criteria.append(InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30))
        criteria.append(
            ActorSpeedAboveThresholdTest(
                self.ego_vehicles[0],
                speed_threshold=0.1,
                below_threshold_max_time=180.0,
                terminate_on_failure=True,
                name="AgentBlockedTest",
            )
        )

        return criteria