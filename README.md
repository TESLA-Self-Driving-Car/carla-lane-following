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

## Prerequisites

Download the latest [CARLA release](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Dev/CARLA_Latest.tar.gz).

Say you unzip to ~/carla-release

git clone git@github.com:xpharry/carla-lane-following.git -b develop

put following line in your ~/.bashrc file

export PYTHONPATH=$PYTHONPATH:~/carla-release/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg

## Experiments

**Test the Execution of waypoint following. Make sure pygame is installed.**

Terminal 1:

run:

```
./CarlaUE4.sh /Game/Carla/Maps/Town05
```

Terminal 2:

run:

```
python code/generate_waypoints_with_autopilot.py
```

then run:

```
python code/waypoint_follower_01.py --control-method MPC
```
