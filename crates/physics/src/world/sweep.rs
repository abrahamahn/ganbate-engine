use super::*;

impl PhysicsWorld {
    /// Wakes a body: a dynamic one with its island (see [`Self::wake_islands`]), a kinematic one
    /// alone; an awake one restarts its sleep timer.
    pub(super) fn wake_body(&mut self, b: u32) {
        self.wake_islands(&[b]);
    }

    /// Wakes `roots` and, for each dynamic one that was asleep, every sleeping dynamic body linked
    /// to it by touching contacts and joints, transitively: its island. Kinematic bodies wake
    /// alone and pass nothing on, as islands leave them out: they are never pushed, so bodies
    /// resting on the same platform are not linked through it. Awake roots restart their sleep
    /// timers. Each woken body's own contact lists and joints give its links, so the cost is
    /// that of the island, not of the world.
    pub(super) fn wake_islands(&mut self, roots: &[u32]) {
        let mut stack: Vec<u32> = Vec::new();
        for &b in roots {
            let body = &mut self.bodies[b as usize];
            if !body.alive || body.kind == BodyKind::Static {
                continue;
            }
            body.sleep_time = 0.0;
            if body.awake {
                continue;
            }
            if body.is_dynamic() {
                stack.push(b);
            } else {
                body.awake = true;
                self.set_awake_bit(b, true);
            }
        }
        let asleep = |b: &Body| b.alive && !b.awake && b.is_dynamic();
        let mut near: Vec<u32> = Vec::new();
        while let Some(x) = stack.pop() {
            let body = &mut self.bodies[x as usize];
            if body.awake {
                continue;
            }
            body.awake = true;
            body.sleep_time = 0.0;
            self.set_awake_bit(x, true);
            near.clear();
            self.push_neighbours(x, &mut near);
            stack.extend(
                near.iter()
                    .copied()
                    .filter(|&o| asleep(&self.bodies[o as usize])),
            );
        }
    }

    /// Appends the bodies touching collider `c` through touching solid contacts (repeats
    /// included).
    pub(super) fn push_touching(&self, c: u32, out: &mut Vec<u32>) {
        for &k in &self.colliders[c as usize].contacts {
            let ct = &self.contacts[k as usize];
            if ct.touching && !ct.sensor {
                out.push(self.colliders[ct.other(c) as usize].body);
            }
        }
    }

    /// Appends the bodies touching body `b` or jointed to it (repeats included).
    pub(super) fn push_neighbours(&self, b: u32, out: &mut Vec<u32>) {
        for &c in &self.bodies[b as usize].colliders {
            self.push_touching(c, out);
        }
        for &j in &self.bodies[b as usize].joints {
            let d = &self.joints[j as usize].desc;
            for o in [d.body_a.index, d.body_b.index] {
                if o != b {
                    out.push(o);
                }
            }
        }
    }

    /// Wakes the dynamic bodies touching `b` or jointed to it (before `b` goes away, moves or
    /// changes shape), with their islands.
    pub(super) fn wake_neighbours(&mut self, b: u32) {
        let mut others = Vec::new();
        self.push_neighbours(b, &mut others);
        others.retain(|&o| self.bodies[o as usize].is_dynamic());
        self.wake_islands(&others);
    }

    /// Wakes the dynamic bodies touching collider `c` (before it goes away), with their islands;
    /// what touches the body's other colliders is not disturbed (a level losing one of its
    /// pieces wakes only what rested on that piece).
    pub(super) fn wake_touching_collider(&mut self, c: u32) {
        let mut others = Vec::new();
        self.push_touching(c, &mut others);
        others.retain(|&o| self.bodies[o as usize].is_dynamic());
        self.wake_islands(&others);
    }

    /// Wakes the sleeping dynamic bodies (with their islands) whose colliders' broad-phase boxes
    /// overlap those of `colliders` and that would collide with them: what static geometry moved
    /// or grown into them must react, as static bodies never wake anything through contacts
    /// (whose pairs with sleeping bodies are not even updated).
    pub(super) fn wake_overlapping(&mut self, colliders: &[u32]) {
        let mut found: Vec<u32> = Vec::new();
        for &c in colliders {
            let col = &self.colliders[c as usize];
            self.moving_tree.query(&col.fat, &mut |_, o| {
                let ob = self.colliders[o as usize].body;
                let other = &self.bodies[ob as usize];
                if ob != col.body && other.is_dynamic() && !other.awake && self.should_collide(c, o)
                {
                    found.push(ob);
                }
                true
            });
        }
        self.wake_islands(&found);
    }

    /// Continuous collision: a dynamic body that moved farther than half its smallest size this
    /// step (or any bullet) is swept from its start position, in its orientation at the end of
    /// the step, against static and kinematic geometry it collides with (the same filter as
    /// contacts, joints included). A translating kinematic collider can strike even a sleeping
    /// body. Its sweep uses relative translation and pushes the body along the surface normal for
    /// the remaining time; fixed geometry still limits that motion.
    ///
    /// Three kinds of hit are not taken, because the body's contacts already handle them and
    /// taking them would stop it for good (also sliding along): a time of impact of zero (the
    /// shape starts touching, or within the cast's tolerance of touching); against one-sided
    /// triangles, a sweep that neither carries the shape's centre across the triangle's plane nor
    /// ends close to it (a body gliding over a flat floor touches the next triangle at every
    /// seam); and against convex geometry, a target that only the shape's outer skin (the linear
    /// slop) beside its path reaches (a body gliding over a floor of flush boxes, sunk into one by
    /// a fraction of a millimetre, meets the next one's side at every seam). For the first kind,
    /// a small ball about the shape's centre (a quarter of the largest that fits inside it, as in
    /// Box2D) is swept instead, so a body already touching a thin wall still cannot pass through
    /// it. When that ball stops the body, the shape is left deep in the surface (an elongated one
    /// whose near end, touching, turned the approach into spin, by up to three quarters of its
    /// inner radius), so its velocity into the surface goes as well and the next step's contacts
    /// only push it out. Whatever the sweeps take, a centre never passes through a one-sided
    /// triangle from its front (see [`centre_through`]): the narrow phase would lose the body
    /// behind it.
    ///
    /// What the sweep does not see: a turn within the step. A thin body spinning fast enough to
    /// swing through a wall and back out of it within one step is not stopped (Box2D sweeps the
    /// rotation as well).
    pub(super) fn continuous(&mut self, moved: &[u32], dt: f32) {
        let kinematics = self.kinematic_sweeps(moved);
        // Keep the same stable body order for solver members and newly reachable sleepers.
        let mut members = moved.iter().copied().peekable();
        let mut sleepers = kinematics.sleepers.iter().copied().peekable();
        let candidates = std::iter::from_fn(|| {
            if sleepers
                .peek()
                .is_some_and(|s| members.peek().is_none_or(|m| s < m))
            {
                sleepers.next().map(|b| (b, false))
            } else {
                members.next().map(|b| (b, true))
            }
        });
        let mut hit_events = None;
        for (b, integrated) in candidates {
            self.sweep_body(b, integrated, dt, &kinematics, &mut hit_events);
        }
    }

    pub(super) fn kinematic_sweeps(&self, moved: &[u32]) -> KinematicSweeps {
        // The normal broad phase indexes the end poses. A translating platform can cross a
        // sleeping body and finish nowhere near it, so index the path of each moving platform.
        let mut kinematics = KinematicSweeps::default();
        let mut search_sleepers = false;
        for &b in moved {
            let body = &self.bodies[b as usize];
            let travel = body.center - body.center0;
            if body.kind != BodyKind::Kinematic || travel == Vec3::ZERO {
                continue;
            }
            kinematics.travel.insert(b, travel);
            for &c in &body.colliders {
                if self.colliders[c as usize].sensor {
                    continue;
                }
                if kinematics.tree.is_empty() {
                    // A scene containing only moving scenery needs no dynamic-response sweeps.
                    let mut dynamics = self
                        .bodies
                        .iter()
                        .filter(|b| b.alive && b.is_dynamic())
                        .peekable();
                    if dynamics.peek().is_none() {
                        return kinematics;
                    }
                    search_sleepers = dynamics.any(|b| !b.awake);
                }
                let swept = self.collider_box(c).sweep(-travel);
                kinematics.tree.insert(swept, c);
                if !search_sleepers {
                    continue;
                }
                // Collect unintegrated bodies at rest in the platform's path. Looking up the
                // sleepers here avoids testing every sleeping collider against a distant lift.
                self.moving_tree
                    .query(&swept.expand(self.settings.linear_slop), &mut |_, o| {
                        let other = &self.colliders[o as usize];
                        let owner = &self.bodies[other.body as usize];
                        if !other.sensor
                            && owner.is_dynamic()
                            && !owner.awake
                            && moved.binary_search(&other.body).is_err()
                        {
                            kinematics.sleepers.insert(other.body);
                        }
                        true
                    });
            }
        }

        kinematics
    }

    pub(super) fn sweep_body(
        &mut self,
        b: u32,
        integrated: bool,
        dt: f32,
        kinematics: &KinematicSweeps,
        hit_events: &mut HitEventIndex,
    ) {
        let body = &self.bodies[b as usize];
        if !body.is_dynamic() || (!body.awake && kinematics.tree.is_empty()) {
            return;
        }
        // A contact can wake a whole island. Only a body integrated in this step refreshed
        // its starting center; the other sleepers have not moved.
        let start = if integrated {
            body.center0
        } else {
            body.center
        };
        let mut path = SweepInterval {
            center: start,
            travel: body.center - start,
            remaining: 1.0,
            after_hit: false,
        };
        let threshold = 0.5 * body.min_extent.max(self.settings.linear_slop);
        let mut check_fixed = body.bullet || path.travel.length() > threshold;
        if !check_fixed && kinematics.tree.is_empty() {
            return;
        }
        let mut velocity = body.linear_velocity;
        let mask = body.translation_mask();
        let mut pushed = false;
        let mut travel_planes = Vec::new();
        let mut speed_planes = Vec::new();
        // Move chronologically. After a moving obstacle pushes the body, sweep its new
        // path as well: the obstacle must not shove it through a second, fixed wall.
        // If crowded between incompatible obstacles, keep the last safe position.
        for _ in 0..8 {
            let Some(hit) = self.first_ccd_hit(b, &path, check_fixed, kinematics) else {
                path.center += path.travel;
                break;
            };
            let impact = hit.impact;
            let fixed = hit.target_travel == Vec3::ZERO;
            if !fixed && path.remaining > 0.0 {
                let approach = (hit.target_travel * path.remaining - path.travel)
                    .dot(impact.normal)
                    / (dt * path.remaining);
                self.report_swept_hit(hit_events, hit, approach);
            }
            path.after_hit = true;
            path.center += path.travel * impact.fraction;
            path.remaining *= 1.0 - impact.fraction;
            if fixed && !pushed {
                // Preserve the fixed-geometry stop. A core hit also cancels the inward
                // velocity so the next step's solver only has to push out the penetration.
                if impact.core {
                    remove_closing(&mut velocity, Vec3::ZERO, impact.normal, mask);
                }
                path.travel = Vec3::ZERO;
                if kinematics.tree.is_empty() {
                    break;
                }
            } else {
                path.travel *= 1.0 - impact.fraction;
            }
            let target = self.colliders[hit.collider as usize].body;
            if let Some(p) = crate::ccd::Plane::new(impact.normal, hit.target_travel, mask) {
                travel_planes.push(p);
            }
            if let Some(p) = crate::ccd::Plane::new(
                impact.normal,
                self.bodies[target as usize].linear_velocity,
                mask,
            ) {
                speed_planes.push(p);
            }
            pushed |= !fixed;
            if pushed {
                check_fixed = true;
                let cap = self.settings.max_linear_speed;
                let travel = if path.remaining > 0.0 {
                    crate::ccd::permitted(path.travel / path.remaining, &travel_planes, cap * dt)
                        .map(|v| v * path.remaining)
                } else {
                    Some(Vec3::ZERO)
                };
                let speed = crate::ccd::permitted(velocity, &speed_planes, cap);
                let (Some(travel), Some(speed)) = (travel, speed) else {
                    // A prescribed platform can overrun a body pinned by scenery, axis
                    // locks or its speed cap. Keep the last safe pose in that case.
                    velocity = Vec3::ZERO;
                    break;
                };
                path.travel = travel;
                velocity = speed;
            }
        }
        if path.after_hit {
            if pushed {
                self.wake_body(b);
            }
            let body = &mut self.bodies[b as usize];
            body.center = path.center;
            body.position = body.center - body.rotation.rotate(body.local_center);
            body.linear_velocity = velocity;
            self.refresh_body_proxies(b, false);
        }
    }

    /// CCD resolves a moving wall's approach after the discrete solver. Keep the strongest
    /// impact per ordered pair from either source; Begin/End still follow the manifold.
    pub(super) fn report_swept_hit(
        &mut self,
        reported: &mut HitEventIndex,
        hit: CcdHit,
        speed: f32,
    ) {
        if !speed.is_finite() || speed <= self.settings.hit_event_threshold {
            return;
        }
        let index = reported.get_or_insert_with(|| {
            self.pending
                .contacts
                .iter()
                .enumerate()
                .filter_map(|(i, event)| match event {
                    ContactEvent::Hit { a, b, .. } => Some(((a.index, b.index), i)),
                    _ => None,
                })
                .collect()
        });
        let (a, b, normal) = if hit.collider < hit.moving {
            (hit.collider, hit.moving, hit.impact.normal)
        } else {
            (hit.moving, hit.collider, -hit.impact.normal)
        };
        let event = ContactEvent::Hit {
            a: self.collider_id(a),
            b: self.collider_id(b),
            point: hit.impact.point,
            normal,
            speed,
        };
        if let Some(&i) = index.get(&(a, b)) {
            if matches!(self.pending.contacts[i], ContactEvent::Hit {speed: old, ..} if old < speed)
            {
                self.pending.contacts[i] = event;
            }
        } else {
            index.insert((a, b), self.pending.contacts.len());
            self.pending.contacts.push(event);
        }
    }

    /// Earliest impact along the remaining interval, measured in the target's frame. We hold
    /// both colliders at their end orientations as in the original fixed-geometry sweep.
    pub(super) fn first_ccd_hit(
        &self,
        b: u32,
        path: &SweepInterval,
        check_fixed: bool,
        kinematics: &KinematicSweeps,
    ) -> Option<CcdHit> {
        let body = &self.bodies[b as usize];
        let slop = self.settings.linear_slop;
        let start = Iso::new(
            path.center - body.rotation.rotate(body.local_center),
            body.rotation,
        );
        let mut best: Option<CcdHit> = None;
        let mut candidates = SmallBuf::<(u32, Vec3), 32>::new((0, Vec3::ZERO));
        for &c in &body.colliders {
            let col = &self.colliders[c as usize];
            if col.sensor {
                continue;
            }
            let Some(shape) = col.shape.as_convex() else {
                continue;
            };
            let from = start.mul(&col.local);
            let swept = col.shape.aabb(&from).sweep(path.travel).expand(slop);
            candidates.clear();
            if check_fixed && path.travel != Vec3::ZERO {
                let mut fixed = |_: u32, o: u32| {
                    let owner = self.colliders[o as usize].body;
                    if !self.bodies[owner as usize].is_dynamic()
                        && !kinematics.travel.contains_key(&owner)
                    {
                        candidates.push((o, Vec3::ZERO));
                    }
                    true
                };
                self.static_tree.query(&swept, &mut fixed);
                self.moving_tree.query(&swept, &mut fixed);
            }
            kinematics.tree.query(&swept, &mut |_, o| {
                let owner = self.colliders[o as usize].body;
                candidates.push((o, kinematics.travel[&owner]));
                true
            });
            if candidates.is_empty() {
                continue;
            }
            let moving = Convex::new(shape, from);
            let center0 = moving.center();
            let core_radius = 0.25 * col.shape.inner_radius();
            let core = Convex::new(
                ConvexShape::Sphere(core_radius),
                Iso::from_position(center0),
            );
            // Shave the skin beside the path so a seam between flush boxes is not a wall.
            let skin = slop.min(0.75 * col.shape.inner_radius());
            let mut shrunk_verts = [Vec3::ZERO; crate::hull::MAX_HULL_VERTICES];
            let shrunk_hull_len = if let ConvexShape::Hull(h) = shape {
                let farthest = h.outer_face_dist;
                let k = if farthest > skin {
                    1.0 - skin / farthest
                } else {
                    1.0
                };
                let n = h.vertices.len();
                for (dst, &v) in shrunk_verts[..n].iter_mut().zip(&h.vertices) {
                    *dst = h.centroid + (v - h.centroid) * k;
                }
                Some((n, h.centroid))
            } else {
                None
            };
            let inner_shape = match shape {
                ConvexShape::Sphere(r) => ConvexShape::Sphere(r - skin),
                ConvexShape::Capsule {
                    half_height,
                    radius,
                } => ConvexShape::Capsule {
                    half_height,
                    radius: radius - skin,
                },
                ConvexShape::Box(h) => ConvexShape::Box(h - Vec3::splat(skin)),
                other => other,
            };
            for &(other, target_travel) in candidates.as_slice() {
                let oc = &self.colliders[other as usize];
                if oc.sensor || !self.should_collide(c, other) {
                    continue;
                }
                let remaining_target = target_travel * path.remaining;
                let relative = path.travel - remaining_target;
                let (direction, distance) = relative.normalize_and_length();
                if distance == 0.0
                    || (!body.bullet && !check_fixed && distance <= 0.5 * body.min_extent.max(slop))
                {
                    continue;
                }
                let mut at = self.collider_iso(other);
                at.position -= remaining_target;
                let mut take = |impact: Option<Impact>| {
                    let Some(impact) = impact else { return };
                    if impact.fraction >= 1.0
                        || (target_travel != Vec3::ZERO
                            && impact
                                .normal
                                .mul_elem(body.translation_mask())
                                .length_squared()
                                < 1e-12)
                    {
                        return;
                    }
                    if best.is_none_or(|old| {
                        impact.fraction < old.impact.fraction
                            || (impact.fraction == old.impact.fraction && other < old.collider)
                    }) {
                        best = Some(CcdHit {
                            impact: Impact {
                                // The relative cast fixed B at its interval-start pose.
                                // Bring its surface witness forward to the instant of contact.
                                point: impact.point + remaining_target * impact.fraction,
                                ..impact
                            },
                            moving: c,
                            collider: other,
                            target_travel,
                        });
                    }
                };
                let mut triangle = |t: &TriangleRef| {
                    let (n, v0) = (at.vector(t.normal()), at.point(t.vertices[0]));
                    let (s0, s1) = (n.dot(center0 - v0), n.dot((center0 + relative) - v0));
                    if s0 >= 0.0 && (s0 - s1 >= core_radius || s1 <= core_radius) {
                        let target = Convex::new(ConvexShape::Triangle(t.vertices), at);
                        take(time_of_impact(
                            &moving,
                            &core,
                            relative,
                            &target,
                            slop,
                            path.after_hit,
                        ));
                        take(centre_through(
                            t,
                            &at,
                            center0,
                            relative,
                            core_radius + slop,
                        ));
                    }
                    true
                };
                match &oc.shape {
                    Shape::Mesh(m) => {
                        let local = crate::shape::transform_aabb(&swept, &at.inverse());
                        m.query_aabb(&local, &mut triangle);
                    }
                    Shape::Heightfield(h) => {
                        let local = crate::shape::transform_aabb(&swept, &at.inverse());
                        h.query_aabb(&local, &mut triangle);
                    }
                    other_shape => {
                        if let Some(os) = other_shape.as_convex() {
                            let target = Convex::new(os, at);
                            let front =
                                |s: &Convex| s.support_core(direction).dot(direction) + s.radius();
                            let inner_hit = if let Some((n, centroid)) = shrunk_hull_len {
                                let verts = &shrunk_verts[..n];
                                let local_d = from.inverse_vector(direction);
                                let mut best_i = 0;
                                let mut best_d = f32::NEG_INFINITY;
                                for (i, v) in verts.iter().enumerate() {
                                    let p = v.dot(local_d);
                                    if p > best_d {
                                        best_d = p;
                                        best_i = i;
                                    }
                                }
                                let inner_front = from.point(verts[best_i]).dot(direction);
                                let ahead = direction * (front(&moving) - inner_front);
                                gjk::cast_hull_vertices(
                                    verts,
                                    centroid,
                                    Iso::new(from.position + ahead, from.rotation),
                                    relative,
                                    &target,
                                    0.0,
                                )
                                .is_some()
                            } else {
                                let ahead = direction
                                    * (front(&moving) - front(&Convex::new(inner_shape, from)));
                                let inner = Convex::new(
                                    inner_shape,
                                    Iso::new(from.position + ahead, from.rotation),
                                );
                                gjk::cast(&inner, relative, &target, 0.0).is_some()
                            };
                            if inner_hit {
                                take(time_of_impact(
                                    &moving,
                                    &core,
                                    relative,
                                    &target,
                                    slop,
                                    path.after_hit,
                                ));
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// A hash of every body's pose and velocity (bit patterns, index order): equal hashes after
    /// equal inputs show the simulation is deterministic.
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |x: f32| {
            for byte in x.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for b in self.bodies.iter().filter(|b| b.alive) {
            for v in [b.position, b.linear_velocity, b.angular_velocity] {
                eat(v.x);
                eat(v.y);
                eat(v.z);
            }
            for q in b.rotation.to_array() {
                eat(q);
            }
        }
        h
    }
}
