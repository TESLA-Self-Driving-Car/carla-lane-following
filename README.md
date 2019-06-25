# Carla Lane Following

MPC, Stanley and PurePursuit

## Basic operations

1. Client

```
# First of all, we need to create the client that will send the requests
# to the simulator. Here we'll assume the simulator is accepting
# requests in the localhost at port 2000.
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
```

2. World

```
# Once we have a client we can retrieve the world that is currently
# running.
world = client.get_world()
```

3. Map

A map is obtained from the world.

```
map = world.get_map()
```

4. Location and Transform

It is natural to think that "Location" and "Transform" object can be initiated as follows,

```
example_location = carla.Location(x=0, y=0)
example_transform = carla.Transform(carla.Location(x=0, y=0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0))
```

which is also successful.

But for "Waypoint", it is not.

5. Waypoint

The way to construct a waypoint from values is like follows.

```
map = world.get_map()
example_location = carla.Location(x=-10, y=10)
example_waypoint = map.get_waypoint(example_location) # map is predefined
```

The reason of this is that "Waypoint" could not be valid or meaningful without a predefined "Map" object.

## How to run

Terminal 1:

```

```

Terminal 2:

```

```
