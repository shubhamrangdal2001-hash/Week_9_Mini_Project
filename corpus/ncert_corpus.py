"""
ncert_corpus.py
────────────────
Synthetic text that mirrors the content and structure of the five
NCERT Class 9 Science chapters on Physics (Chapters 8–12).

The text is modelled on the real textbook:
  Chapter 8  — Motion
  Chapter 9  — Force and Laws of Motion
  Chapter 10 — Gravitation
  Chapter 11 — Work and Energy
  Chapter 12 — Sound

ALL_CHAPTERS is a dict of {chapter_name: raw_text}.
The raw_text deliberately contains:
  • Typical PDF extraction artefacts (fused words, extra blank lines)
  • Section headers in the NCERT numbering style ("8.1 DESCRIBING MOTION")
  • Worked examples ("EXAMPLE 8.1")
  • Exercise questions ("1. An athlete …")
  • Equations and figure references ("[Fig. 8.1: …]")
"""

CH8 = """
Chapter 8: Motion

8.1 DESCRIBING MOTION

Motion is one of the most common phenomena we observe in everyday life.
A car moving on a road, a bird flying in the sky, a ball rolling on the
ground — all these are examples of motion. In this chapter, we shall
describe the motion of objects along a straight line.

An object is said to be in motion if it changes its position with time
with respect to some fixed reference point. If the position of an object
does not change with time, the object is said to be at rest.

SCIENCE

8.2 UNIFORM MOTION AND NON-UNIFORM MOTION

If an object moves equal distances in equal intervals of time, the object
is said to be in uniform motion. A car travelling on a straight highway at
a constant speed of 60 km/h covers equal distances in equal time intervals.

If an object does not cover equal distances in equal intervals of time, the
motion is said to be non-uniform motion. A ball rolling down a slope or a
car in heavy traffic shows non-uniform motion.

8.3 MEASURING THE RATE OF MOTION

The distance travelled by an object in unit time is called its speed.
Speed = Distance / Time
SI unit of speed is metre per second (m s-1 or ms-1).

Average speed = Total distance / Total time

EXAMPLE 8.1
An object travels 16 m in 4 s and then another 16 m in 2 s. What is the
average speed of the object?

Solution:
Total distance = 16 + 16 = 32 m
Total time = 4 + 2 = 6 s
Average speed = 32 / 6 = 5.33 m s-1

Therefore the average speed of the object is 5.33 m s-1.

8.3.1 Speed with Direction — Velocity

Velocity is the speed of an object in a specified direction. It is a vector
quantity. The SI unit of velocity is also m s-1.

Velocity = Displacement / Time

If the velocity of an object changes, either the speed changes or the
direction changes. An object moving in a circular path at constant speed has
a changing velocity because the direction keeps changing.

8.4 RATE OF CHANGE OF VELOCITY — ACCELERATION

The rate of change of velocity of an object with time is called acceleration.
Acceleration = (Final velocity - Initial velocity) / Time
             = (v - u) / t

The SI unit of acceleration is m s-2 (metre per second squared).

When velocity increases with time, acceleration is positive.
When velocity decreases with time, acceleration is negative (deceleration or retardation).
If velocity is constant, acceleration is zero (uniform motion).

EXAMPLE 8.2
A car starts from rest and reaches a velocity of 20 m s-1 in 10 s.
What is the acceleration of the car?

Solution:
u = 0 (starts from rest)
v = 20 m s-1
t = 10 s
a = (v - u) / t = (20 - 0) / 10 = 2 m s-2

Therefore the acceleration of the car is 2 m s-2.

8.5 GRAPHICAL REPRESENTATION OF MOTION

8.5.1 Distance-Time Graph

A distance-time graph shows how distance changes with time.
[Fig. 8.1: Distance-time graph for uniform motion — a straight line]

The slope of the distance-time graph gives the speed of the object.
For uniform motion the graph is a straight line. A steeper slope means
greater speed.

8.5.2 Velocity-Time Graph

A velocity-time graph shows how velocity changes with time.
[Fig. 8.2: Velocity-time graph showing uniform acceleration]

The slope of the velocity-time graph gives the acceleration.
The area under the velocity-time graph gives the displacement.

8.6 EQUATIONS OF MOTION FOR UNIFORM ACCELERATION

When an object moves along a straight line with uniform acceleration, its
motion can be described by three equations (equations of motion):

1. v = u + at
2. s = ut + (1/2) at2
3. v2 = u2 + 2as

where:
  u = initial velocity (m s-1)
  v = final velocity (m s-1)
  a = acceleration (m s-2)
  t = time (s)
  s = displacement (m)

EXAMPLE 8.3
A ball is thrown vertically upward with an initial velocity of 19.6 m s-1.
Find the maximum height reached.

Solution:
At maximum height, final velocity v = 0
u = 19.6 m s-1, a = -9.8 m s-2 (gravity acts downward)
Using v2 = u2 + 2as:
0 = (19.6)2 + 2 × (-9.8) × s
19.6 s = (19.6)2 / (2 × 9.8) = 384.16 / 19.6 = 19.6 m

Therefore the maximum height reached is 19.6 m.

8.7 UNIFORM CIRCULAR MOTION

When an object moves in a circular path at constant speed, it is said to
be in uniform circular motion. Although the speed is constant, the velocity
continuously changes direction. Hence uniform circular motion is an
accelerated motion.

The acceleration in uniform circular motion is directed toward the centre
of the circle and is called centripetal acceleration.

EXERCISES

1. An athlete completes one round of a circular track of diameter 200 m in
   40 s. What will be the distance covered and the displacement at the end
   of 2 minutes 20 s?

2. Joseph jogs from one end A to the other end B of a straight 300 m road
   in 2 minutes 30 s and then turns around and jogs 100 m back to point C
   in another 1 minute. Find average speed and average velocity.

3. Abdul while driving to school, computes the average speed for his trip
   to be 20 km/h. On his return trip along the same route, there is less
   traffic and the average speed is 30 km/h. What is the average speed for
   Abdul's trip?
"""

CH9 = """
Chapter 9: Force and Laws of Motion

9.1 BALANCED AND UNBALANCED FORCES

A force is a push or pull on an object. Forces can change the state of
motion of an object — they can make a stationary object move, a moving
object stop, change the speed or direction of a moving object, or change
the shape of an object.

When two forces act on an object in opposite directions and are equal in
magnitude, the net force is zero. These are called balanced forces.
Balanced forces cannot change the state of rest or motion of an object.

When the net force on an object is not zero, the forces are unbalanced.
An unbalanced force changes the state of motion of an object.

9.2 FIRST LAW OF MOTION — LAW OF INERTIA

Newton's First Law of Motion: An object remains in a state of rest, or
in a state of uniform motion in a straight line, unless compelled to change
that state by an applied force.

This tendency of objects to resist change in their state of motion is
called inertia. Mass is a measure of inertia — a more massive object
has greater inertia and requires a larger force to change its state of motion.

Examples of inertia:
• When a bus suddenly stops, passengers tend to lurch forward because
  their bodies tend to continue in the state of motion.
• Dust comes out of a carpet when it is beaten because the carpet moves
  but the dust particles tend to remain at rest due to inertia.
• A coin placed on a card over a glass falls into the glass when the card
  is flicked quickly because the coin remains at rest while the card moves.

9.3 SECOND LAW OF MOTION

The momentum of an object is the product of its mass and velocity:
  p = m × v

The SI unit of momentum is kg m s-1.

Newton's Second Law of Motion: The rate of change of momentum of an object
is proportional to the applied unbalanced force in the direction of the force.

Mathematically:
  F = ma

where F = applied force (N or kg m s-2)
      m = mass of the object (kg)
      a = acceleration (m s-2)

One Newton is defined as the force that produces an acceleration of 1 m s-2
in an object of mass 1 kg. So 1 N = 1 kg m s-2.

EXAMPLE 9.1
A cricket ball of mass 0.15 kg is moving with a velocity of 12 m s-1.
A batsman hits it back in the opposite direction with a velocity of 20 m s-1.
If the bat is in contact for 0.01 s, find the force applied.

Solution:
Change in momentum = m(v - u) = 0.15 × (20 - (-12)) = 0.15 × 32 = 4.8 kg m s-1
Force = Change in momentum / Time = 4.8 / 0.01 = 480 N

Therefore the force applied by the batsman is 480 N.

9.4 THIRD LAW OF MOTION

Newton's Third Law of Motion: For every action there is an equal and opposite
reaction; the forces always act on two different objects.

When you push a wall with force F, the wall pushes back on you with the same
force F in the opposite direction. The action and reaction forces are equal
in magnitude but opposite in direction, and they always act on different objects.

Examples of Newton's Third Law:
• A gun recoils when a bullet is fired. The bullet exerts an equal and
  opposite force on the gun.
• A rocket moves forward by expelling gases backward.
• Swimming: a swimmer pushes the water backward, the water pushes the
  swimmer forward.

9.5 CONSERVATION OF MOMENTUM

The total momentum of a system of objects is conserved (remains constant)
if no external unbalanced force acts on the system.

For a two-body system:
  m1 u1 + m2 u2 = m1 v1 + m2 v2

EXAMPLE 9.2
A bullet of mass 20 g (0.020 kg) is fired from a gun of mass 4 kg at a
velocity of 400 m s-1. Find the recoil velocity of the gun.

Solution:
Before firing: total momentum = 0 (both at rest)
After firing: 0.020 × 400 + 4 × v_gun = 0
8 + 4 × v_gun = 0
v_gun = -8 / 4 = -2 m s-1

Therefore the gun recoils at 2 m s-1 in the direction opposite to the bullet.

EXERCISES

1. An object of mass 100 g is moving with a velocity of 10 m s-1. A force
   of 1 N acts on it for 5 s in the direction of motion. Find the final
   momentum.

2. Using a horizontal force of 200 N, we intend to move a wooden cabinet
   across a floor at a constant velocity. What is the friction force that
   will be exerted on the cabinet?

3. Two objects each of mass 1.5 kg are moving in the same direction with
   speeds 2 m s-1 and 3 m s-1 respectively. They collide and stick together.
   Find the final velocity.
"""

CH10 = """
Chapter 10: Gravitation

10.1 GRAVITATION

Every object in the universe attracts every other object with a force which
is called the gravitational force. This was first recognised by Sir Isaac Newton.

10.1.1 Universal Law of Gravitation

Newton's Universal Law of Gravitation: Every object in the universe attracts
every other object with a force that is directly proportional to the product
of their masses and inversely proportional to the square of the distance
between them.

Mathematically:
  F = G × m1 × m2 / d2

where:
  F  = gravitational force (N)
  m1 = mass of first object (kg)
  m2 = mass of second object (kg)
  d  = distance between the centres of the two objects (m)
  G  = Universal Gravitational Constant = 6.673 × 10-11 N m2 kg-2

[Fig. 10.1: Gravitational force between two masses m1 and m2 at distance d]

The value G = 6.673 × 10-11 N m2 kg-2 was determined experimentally by
Henry Cavendish in 1798.

10.2 FREE FALL

When an object falls under the influence of gravity alone (without air
resistance), it is called free fall. During free fall, the only force
acting on the object is gravity.

The acceleration produced during free fall is called the acceleration due
to gravity, denoted by g.

g = GM / R2

where M = mass of the Earth = 6 × 1024 kg
      R = radius of the Earth = 6.4 × 106 m
      G = 6.673 × 10-11 N m2 kg-2

Substituting: g = 9.8 m s-2 (approximately 10 m s-2)

g is taken as positive when an object falls toward the Earth.
g is taken as negative when an object is thrown upward.

EXAMPLE 10.1
An object of mass 2 kg is dropped from a height. How far does it fall in 3 s?

Solution:
u = 0 (dropped from rest), a = g = 9.8 m s-2, t = 3 s
s = ut + (1/2)gt2 = 0 + (1/2) × 9.8 × 9 = 44.1 m

Therefore the object falls 44.1 m in 3 s.

10.3 MASS AND WEIGHT

Mass is the amount of matter contained in an object. It is a scalar quantity
measured in kilograms (kg). The mass of an object remains constant everywhere
in the universe.

Weight is the force with which an object is attracted toward the Earth.
  W = m × g

Weight is a vector quantity whose SI unit is Newton (N).
Since g varies from place to place, weight varies. At the poles, g is slightly
greater than at the equator.

10.3.1 Weight on the Moon

The mass of the Moon is about 1/100 of Earth's mass, and its radius is about
1/4 of Earth's radius. The acceleration due to gravity on the Moon is:
  g_Moon = g_Earth / 6 = 9.8 / 6 ≈ 1.63 m s-2

Therefore an object weighs 1/6 of its Earth weight on the Moon.

EXAMPLE 10.2
An object has a mass of 10 kg on Earth. What is its weight on the Moon?
(g_moon = 1.63 m s-2)

Solution:
Weight on Earth = m × g = 10 × 9.8 = 98 N
Weight on Moon  = m × g_moon = 10 × 1.63 = 16.3 N

Therefore the weight of the object on the Moon is 16.3 N.

10.4 THRUST AND PRESSURE

Thrust is the force acting on an object perpendicular to its surface.
Pressure is defined as thrust per unit area.

  Pressure = Force / Area = F / A

SI unit of pressure is Pascal (Pa).
1 Pa = 1 N m-2

Pressure increases with depth in a fluid:
  P = hρg

where h = depth below the surface (m)
      ρ = density of the fluid (kg m-3)
      g = acceleration due to gravity (m s-2)

10.5 BUOYANCY AND ARCHIMEDES' PRINCIPLE

When an object is placed in a fluid, it experiences an upward force called
buoyant force or upthrust. This is because fluid pressure increases with
depth, so the pressure on the bottom of the object is greater than the
pressure on its top.

Archimedes' Principle: When an object is immersed in a fluid, the buoyant
force acting on it is equal to the weight of the fluid displaced by the object.

Buoyant force = Weight of fluid displaced = V × ρ_fluid × g

An object floats if its average density is less than the density of the fluid.
An object sinks if its average density is greater than the density of the fluid.

This explains why:
• A ship made of steel floats — it is hollow and its average density is
  less than water.
• A stone sinks — its density is greater than water.

[Fig. 10.2: Buoyant force on an object immersed in a fluid]

EXERCISES

1. How does the force of gravitation between two objects change when the
   distance between them is reduced to half?

2. A stone is dropped from the edge of a roof. It passes a window 5 m below
   the roof 0.5 s after being dropped. How long does it take to fall another
   5 m to reach the ground?

3. Calculate the weight of a body of mass 5 kg on the surface of the Moon.
"""

CH11 = """
Chapter 11: Work and Energy

11.1 WORK

Work is said to be done when a force acts on an object and causes it to
move in the direction of the force.

Work = Force × Displacement
     = F × s × cos(θ)

where θ is the angle between the force and displacement.

When θ = 0° (force and displacement in same direction):  W = F × s
When θ = 90° (force perpendicular to displacement):      W = 0
When θ = 180° (force opposite to displacement):          W = -F × s

The SI unit of work is Joule (J). 1 J = 1 N × 1 m = 1 N m.

Work is a scalar quantity.

EXAMPLE 11.1
A force of 5 N acts on an object which moves a distance of 2 m in the
direction of the force. Find the work done.

Solution:
W = F × s = 5 × 2 = 10 J

Therefore the work done is 10 J.

11.2 ENERGY

Energy is the capacity to do work. An object that has energy can exert a
force on another object and do work on it.

The SI unit of energy is Joule (J), the same as work.

11.2.1 Kinetic Energy

The energy possessed by an object due to its motion is called kinetic energy.

Kinetic Energy KE = (1/2) × m × v2

where m = mass of the object (kg)
      v = velocity of the object (m s-1)

EXAMPLE 11.2
Find the kinetic energy of an object of mass 15 kg moving with a velocity
of 4 m s-1.

Solution:
KE = (1/2) × m × v2 = (1/2) × 15 × 42 = (1/2) × 15 × 16 = 120 J

Therefore the kinetic energy of the object is 120 J.

11.2.2 Potential Energy

The energy possessed by an object by virtue of its position or configuration
is called potential energy.

Gravitational Potential Energy:
  PE = m × g × h

where m = mass (kg), g = 9.8 m s-2, h = height above the reference level (m)

EXAMPLE 11.3
A ball of mass 2 kg is at a height of 5 m above the ground. Find its potential
energy. (g = 10 m s-2)

Solution:
PE = mgh = 2 × 10 × 5 = 100 J

Therefore the potential energy of the ball is 100 J.

11.3 LAW OF CONSERVATION OF ENERGY

Energy can neither be created nor destroyed; it can only be converted from
one form to another. The total energy of an isolated system remains constant.

During the free fall of an object:
  At the top:       PE = maximum, KE = 0
  While falling:    PE decreases, KE increases
  At the bottom:    PE = 0, KE = maximum
  At every point:   PE + KE = constant

This is the Law of Conservation of Mechanical Energy.

EXAMPLE 11.4
An object of mass 1 kg is dropped from a height of 10 m. Find its kinetic
energy when it has fallen 5 m. (g = 10 m s-2)

Solution:
Initial PE at height 10 m = mgh = 1 × 10 × 10 = 100 J; KE = 0
After falling 5 m: height above ground = 5 m
PE = mgh = 1 × 10 × 5 = 50 J
By conservation: KE = Total energy - PE = 100 - 50 = 50 J

Therefore the kinetic energy after falling 5 m is 50 J.

11.4 POWER

Power is the rate of doing work, i.e., work done per unit time.

  Power = Work done / Time = W / t

The SI unit of power is Watt (W). 1 W = 1 J s-1.

1 kilowatt (kW) = 1000 W
1 horsepower (hp) = 746 W

EXAMPLE 11.5
A lamp consumes 1000 J of electrical energy in 10 s. Find its power.

Solution:
P = W / t = 1000 / 10 = 100 W

Therefore the power of the lamp is 100 W.

11.4.1 Commercial Unit of Energy

The commercial unit of energy is kilowatt-hour (kWh).

1 kWh = 1 kW × 1 hour = 1000 W × 3600 s = 3.6 × 106 J

1 kWh is also called 1 unit of electrical energy.

EXERCISES

1. A body of mass 4 kg is moving with a velocity of 3 m s-1. Find its kinetic
   energy.

2. A man does 4 kJ of work in 50 s. Find his power.

3. Find the energy in kWh consumed in 10 hours by four devices of power 500 W each.

4. An object of mass 40 kg is raised to a height of 5 m above the ground.
   What is its potential energy? (g = 10 m s-2)
"""

CH12 = """
Chapter 12: Sound

12.1 PRODUCTION OF SOUND

Sound is produced by vibration. When an object vibrates, it causes the
surrounding medium (air, water, solid) to vibrate as well, and these
vibrations travel as sound waves.

Examples of sound production:
• A tuning fork vibrates when struck, producing sound.
• Guitar strings vibrate when plucked.
• Vocal cords vibrate when we speak.

Sound requires a medium to travel. It cannot travel through vacuum.
This is why there is no sound in outer space.

12.2 PROPAGATION OF SOUND

Sound travels in the form of longitudinal waves — the particles of the
medium vibrate in the direction of propagation of the wave.

When sound travels through air:
• Compressions: regions where air particles are pushed together (high pressure)
• Rarefactions: regions where air particles are spread apart (low pressure)

[Fig. 12.1: Compressions and rarefactions in a longitudinal wave]

12.3 CHARACTERISTICS OF SOUND

12.3.1 Frequency and Pitch

Frequency (f): the number of complete vibrations (cycles) per second.
SI unit: Hertz (Hz). 1 Hz = 1 vibration per second.

Pitch: the property of sound that depends on its frequency.
A high-frequency sound has a high pitch (shrill). A low-frequency sound
has a low pitch (deep).

The human ear can hear sound in the frequency range 20 Hz to 20 000 Hz
(20 kHz). This is called the audible range.

Infrasound: frequency below 20 Hz (cannot be heard by humans)
Ultrasound: frequency above 20 000 Hz (20 kHz) (cannot be heard by humans)

12.3.2 Amplitude and Loudness

Amplitude: the maximum displacement of the vibrating particle from its
mean (equilibrium) position.

Loudness: the property of sound related to its amplitude.
A large-amplitude sound is loud. A small-amplitude sound is soft.

The loudness of sound is measured in decibels (dB).

12.3.3 Speed of Sound

The speed of sound depends on the medium:
  Speed in air at 25°C  ≈ 346 m s-1
  Speed in water        ≈ 1500 m s-1
  Speed in steel        ≈ 5100 m s-1

Sound travels faster in solids than in liquids, and faster in liquids than in gases.
Sound also travels faster at higher temperatures.

v = f × λ

where v = speed of sound (m s-1)
      f = frequency (Hz)
      λ = wavelength (m)

12.4 REFLECTION OF SOUND — ECHO

Sound, like light, gets reflected when it hits a hard surface. The reflected
sound is called an echo.

For an echo to be heard distinctly, the minimum distance between the source
and the reflecting surface must be 17.2 m (approximately 17 m). This is
because the human ear can distinguish two sounds that are at least 0.1 s apart,
and sound must travel 2 × 17.2 = 34.4 m in 0.1 s.

Formula to find distance using echo:
  d = v × t / 2

where d = distance to the reflecting surface (m)
      v = speed of sound (m s-1)
      t = time for echo to return (s)

EXAMPLE 12.1
A person stands 85 m from a wall and shouts. He hears the echo after some
time. What is the time gap? (Speed of sound = 340 m s-1)

Solution:
Total distance = 2 × 85 = 170 m
Time = Distance / Speed = 170 / 340 = 0.5 s

Therefore the echo is heard 0.5 s after the shout.

12.5 USES OF MULTIPLE REFLECTIONS OF SOUND

Reverberation: The persistence of sound in a large hall due to repeated
reflections from the walls, ceiling and floor. Excessive reverberation makes
speech difficult to understand. Concert halls use sound-absorbing materials
on walls to reduce reverberation.

Stethoscope: Multiple reflections of sound inside a tube allow doctors
to listen to the sounds of the heart and lungs.

Megaphone/horn/loudspeaker: Shape is designed to direct sound by reflection.

12.6 ULTRASOUND

Ultrasound has frequencies above 20 000 Hz (20 kHz). It has many applications:

1. Medical imaging (ultrasonography): Ultrasound pulses are sent into the body
   and the reflected echoes are used to form images of internal organs.
   It is safe — no radiation hazard.

2. SONAR (Sound Navigation And Ranging): Uses ultrasound to find depth of
   the ocean or detect underwater objects (submarines, fish shoals, icebergs).

3. Detecting cracks in metal objects: Ultrasound passes through metal; cracks
   cause partial reflection that can be detected.

4. Cleaning of parts in hard-to-reach places: Objects are placed in a liquid
   and ultrasound waves cause dirt to vibrate off.

5. Echolocation: Bats and dolphins use ultrasound to navigate and hunt.

12.6.1 SONAR Calculations

SONAR sends ultrasound pulses downward and measures the time for the echo
to return. Distance is calculated as:
  d = v × t / 2

EXAMPLE 12.2
A SONAR device on a submarine sends ultrasound pulses and receives the echo
in 4 s. Speed of sound in seawater = 1500 m s-1. Find the depth of the ocean.

Solution:
d = v × t / 2 = 1500 × 4 / 2 = 3000 m

Therefore the depth of the ocean floor is 3000 m.

12.7 STRUCTURE OF THE HUMAN EAR

The outer ear (pinna) collects sound waves and directs them into the ear canal
toward the eardrum (tympanic membrane). The eardrum vibrates with the sound.
These vibrations are transmitted by three tiny bones (ossicles: malleus, incus,
stapes) to the inner ear (cochlea). The cochlea converts vibrations to
electrical nerve signals sent to the brain.

EXERCISES

1. A man is standing at a distance of 680 m from a cliff. He shouts and hears
   the echo after 4 s. Find the speed of sound.

2. What is the frequency range of audible sounds in humans?

3. An echo is heard 0.6 s after a sound is produced. If speed of sound is
   340 m s-1, how far is the reflecting surface?

4. Why is the speed of sound greater in solids than in gases?

5. Calculate the wavelength of sound in air if its frequency is 220 Hz and
   speed is 346 m s-1.
"""

# ── The main export ────────────────────────────────────────────
ALL_CHAPTERS = {
    "Chapter 8: Motion":                 CH8,
    "Chapter 9: Force and Laws of Motion": CH9,
    "Chapter 10: Gravitation":           CH10,
    "Chapter 11: Work and Energy":       CH11,
    "Chapter 12: Sound":                 CH12,
}
