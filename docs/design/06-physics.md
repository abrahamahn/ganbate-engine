# Physics

Ganbate implements rigid-body physics in `crates/physics`, with no dependency beyond `ganbate-core`. A soft-step
solver advances the world in fixed time slices. Native and WebAssembly builds are designed to produce identical
poses and velocities for the same inputs. Authors attach `rigid_body` and `collider` components to scene entities
in Ganbate Studio. The scene cook writes them into the world's entity table.

Games use it for anything that falls, rolls, slides, swings or blocks: dropped props, doors and drawers, characters
walking a town, vehicles later. A game that only needs rays (picking, line of sight, placement on surfaces) uses the
same world's queries without simulating anything.

## 1. What a world holds

| Thing | What it is |
| --- | --- |
| Body | Static (never moves), kinematic (moves only as its velocity or a target says, pushes but is never pushed) or dynamic (moved by gravity, contacts and joints). A pose, velocities, damping, gravity scale, axis locks, sleep state. A dynamic body's mass is kept between 1e-20 and 1e25 kg (its colliders' densities scaled alike beyond that), so momentum stays a number. One free to turn about a single world axis stays on the orientations it can reach, its rest orientation (as created or last teleported) turned about that axis, however long it spins |
| Collider | A shape placed on a body: sphere, capsule, box, convex hull, triangle mesh (static only) or heightfield (static only); density, friction, restitution; layers and a mask; optionally a sensor that reports overlaps without colliding |
| Joint | Ball, hinge (limits, motor, spring), slider (limits, motor, spring), fixed, distance (range, spring), between two bodies' joint frames |
| Contact | A pair of colliders whose broad-phase boxes overlap, with up to one manifold of four points per convex pair (several against meshes) and the impulses carried between steps. Each collider lists its contacts, so teleporting, waking or removing one body costs what it touches, not what the world holds |

Handles carry a generation, so a handle to a destroyed body never reaches whatever reuses its slot.

## 2. One step

1. **Broad phase.** Two dynamic bounding-volume trees, one for static colliders and one for everything that moves.
   A collider's box is its bounds grown by a margin; only colliders that leave their box are re-inserted and look for
   new pairs, which keeps large static levels cheap. Boxes are refreshed again at the end of the step, so queries made
   between steps see every body where it is now.
2. **Narrow phase.** Every pair with an awake moving body gets fresh manifolds: separating-axis tests with face
   clipping for boxes and hulls, closest features for spheres and capsules, per-triangle tests against meshes and
   heightfields with *active edges* so bodies slide over seams of flat floors without catching. A ridge's normals
   stay between its two faces' normals and a vertex's never lean out over a seam, so a gentle crease or a straight
   border is no wall; manifolds of neighbouring triangles with one normal merge and keep the corners of their patch.
   Points come with their separation and are kept up to a speculative distance (2 cm) before contact, so approaching
   bodies stop exactly when they meet instead of overlapping first.
3. **Solver: soft step.** Erin Catto's soft step (Box2D v3, "Solver2D", 2024) in three dimensions: substeps (4 per
   step by default), each integrating velocities, warm-starting every constraint with last step's impulse, solving
   contacts and joints with a soft position correction (a spring of 30 Hz with damping ratio 10 for contacts, stiffer
   against static bodies), integrating positions, then relaxing without the correction. Contact points are found
   once per step; each substep updates their separation from how far the bodies moved and turned (only the core of a
   sphere's or capsule's anchor turns: its touching point stays a radius under the core however it rolls). A
   manifold's points are solved forward and then backward, starting from the other end each substep, so a box landing
   flat does not tip toward whichever corner was solved first. Friction is a cone (`|f| ≤ μ·λn`), restitution is
   applied once after the substeps. Joints turn their anchors with the bodies each substep, and push each body
   through its inverse inertia turned with it too (with the inertia of the start of the step, a very thin rod hinged
   across itself turned about its long axis as if it had not swung, and fell off); contacts keep their anchors and
   the inertia of the start of the step together.
4. **Sleep.** Bodies that stayed still for half a second, together with everything touching or jointed to them (an
   island), stop being simulated until something touches, pushes or moves them. A kinematic body keeps awake what it
   touches or is jointed to only while it moves.
5. **Continuous collision.** A dynamic body that moved farther than half its smallest size in a step, or any body
   marked as a bullet, is swept from its start position (in its orientation at the end of the step) against static
   and kinematic geometry and stopped at the first time of impact, so a small fast ball cannot pass through a thin
   wall. Translating kinematic geometry is swept along its own path relative to each dynamic body. A fast
   platform can wake and push a resting body along its surface normal. A platform moving alongside a bullet at
   the same speed leaves the bullet alone. The rest of a pushed body's path is swept again against the scenery.
   A simultaneous set of moving faces constrains the response together. A step handles up to eight
   successive impacts and keeps the last reached pose if they exhaust it or a prescribed platform cannot be
   followed within the body’s axis locks and maximum speed. A moving-platform impact above the hit threshold
   reports its world-space witness and approach speed, at most once per collider pair in that step. Initial
   contacts are handled by the solver. After an impact, another face reached simultaneously may stop the body
   immediately. A one-sided triangle stops a sweep only when the body's centre crosses it, a convex target
   only when the body's inner shape (the body less a skin of the linear slop, its front kept)
   reaches it, and pairs a joint keeps apart are skipped, so gliding over a mesh floor's seams, or over a floor of
   flush boxes, never sticks, while head-on, off-centre and tip-first hits all stop. A body that starts the step
   touching what it hits is swept as a small ball at its centre instead (a quarter of the largest that fits inside
   it, as in Box2D); when that ball stops it, it also loses its velocity into the surface, so a capsule or plank
   driven into a thin wall at a slant, whose touching end turns the approach into spin, stops at the wall instead of
   passing through. Whatever the sweeps take, a body's centre never passes through a one-sided triangle from its
   front: the narrow phase would lose the body behind it (a plank wedged in a corner of a closed mesh room, pushed
   on by friction, escaped that way). The sweep does not follow turns within a step: a thin body spinning fast
   enough to swing through a wall and back out in one step is not stopped.

## 3. Determinism

Bodies store poses and velocities in `f32`. The GJK distance solver uses `f64` for its simplex so small gaps
beside large faces survive rounding. Speed limits can also use `f64` when squared magnitudes exceed the usable
`f32` range. Both paths use IEEE addition, subtraction, multiplication, division and square root. Angles use
the engine's own polynomial approximations. The solver visits bodies by index and sorts pairs before solving.
It reads neither the clock nor a random source. `PhysicsWorld::state_hash` hashes each body's pose and velocity
so a replay or a test can compare two runs step for step.

## 4. Queries

`raycast` (first hit, with the surface normal and the mesh triangle), `shape_cast` (sweep a convex shape, first time
of impact), `overlap` (every collider a shape intersects) and `contacts_with` (the manifolds a shape would have, for
character controllers and placement tools). All take a filter: layers and mask, a body to ignore, sensors in or out.
A shape cast that starts touching a collider (within a tenth of a millimetre, or overlapping it by at most the
world's `linear_slop`) is stopped by it only when the motion heads into it, so a shape resting on the ground can be
cast along it or up; one that starts deeper inside reports it at once; and the seams a mesh marks inactive answer
with the face normal, as in the narrow phase, so a cast glides over a tessellated floor. Mesh and heightfield
triangles are one-sided for the queries as for contacts and continuous collision: a ray or a shape cast from behind
a triangle passes it, `overlap` reports a shape crossing a mesh whichever side its centre is on, and `contacts_with`
gives the narrow phase's one-sided manifolds (a character pushes itself out of surfaces along them). The character's
own sweeps see triangles from both sides.

## 5. Authoring and runtime

`rigid_body` and `collider` are engine components defined in `crates/scene/src/engine.schema.json`, with
Korean labels. A collider belongs to the nearest rigid body on its entity or an ancestor. With no such body it
becomes static scenery. A child collider forms part of a compound body, and its instances and lamps move with
that body.

Collider sizes use the entity's frame. A reflection stays on one scale axis. When a parent shears a box,
perpendicular drawn edges give an oriented box. Otherwise its collider is the convex hull of the eight drawn
corners. The scene builder reports an error when the sheared box exceeds the supported dimensions. A box
rotated on an unevenly scaled entity without parent shear can retain its authored rotation if that approximation
adds at most 25% to its volume and each shape fits inside the other doubled about its centre. A sphere takes
the largest stretch, and a capsule takes the largest stretch across its axis. The `hull` and `mesh` shapes use
the entity's own mesh geometry.

`ganbate_physics::scene::build` reads the components from the cooked entity table and creates a physics world.
After a step, `ScenePhysics::instance_matrix` gives each instance's updated transform for `setWorldInstance`
on the web or the renderer's native world edits.

## 6. Characters

`Character` is a kinematic capsule a game moves once per step with `move_and_slide(world, displacement, dt)`. It
sweeps the capsule with shape casts and keeps a small skin from every surface; slides along walls and into creases;
treats slopes steeper than its limit (50° by default) as walls; climbs steps up to its step height; snaps to the
ground when walking along or down so it does not bounce down slopes and stairs. Two steep surfaces can together
hold up the capsule, as in a V-shaped channel. An opening approached straight on needs about
`2 × (radius + skin / 8)` plus a few millimetres of clearance. An angled approach can need more room.

Move the character after the world steps. It follows its ground body's translation and rotation when driven by
velocity or a kinematic target. `set_body_pose` teleports a platform and leaves its rider behind. The controller
also pushes dynamic bodies at the point of contact and moves out of surfaces it starts inside. The game supplies
gravity, jumps and walking speed through the displacement it passes each step.

## 7. In the editor and the web library

The web library simulates a cooked world's authored physics: `physicsStart()` copies the bodies built when the world
loaded (`physicsError()` names the entity when they could not be built), `physicsStep(dt, steps)` advances them and
moves the instances and lamps of the bodies that moved, once per call. Each call runs up to 64 steps of at most
1/15 second each. Excess steps are dropped. `physicsStop()` restores the cooked poses, and `physicsHash()` returns
the state hash for comparison with the same steps on a native build. Ganbate Studio's **Simulate** button (P) re-cooks the scene, so the simulation starts from what the viewport
shows, steps at a fixed 60 Hz whatever the display's rate, and stops on a second press.

`ganbate-build simulate WORLD.gbew STEPS` runs the same simulation natively from a cooked world and prints the state
hash and where each moving body ended. The editor's end-to-end test steps the saved sample scene 120 times in the
browser and checks that its hash equals this command's, on WebGPU and on WebGL 2.

## 8. What comes later

| Later | Why | Milestone |
| --- | --- | --- |
| Play in editor with the game's own code on top of the simulation | A game's rules running against the authored world | M3 |
| Joints as components (compound bodies from child entities are done) | Doors on hinges, chains, ragdolls authored in the editor | M3, M6 |
| Vehicles (ray-cast wheels), ragdolls from skeletons | The open-world sample | M9 and later |
| Large-world coordinates (origin shifting per region) | `f32` keeps millimetre precision only within about 10 km of the origin | M9 |
| Multithreaded islands | Thousands of active bodies | M9 |
