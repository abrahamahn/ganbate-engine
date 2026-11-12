use super::*;

impl PhysicsWorld {
    // ---------------------------------------------------------------- the step

    /// Advances the simulation by `dt` seconds (use a fixed step, for example 1/60).
    pub fn step(&mut self, dt: f32) {
        if !(dt > 0.0 && dt.is_finite()) {
            return;
        }
        self.settings = self.settings.sanitized();
        self.steps += 1;
        if self.move_buffer.is_empty() && !self.has_awake_moving_bodies() {
            self.rescale_warm_starts(self.settings.substeps.max(1));
            std::mem::swap(&mut self.events, &mut self.pending);
            self.pending.contacts.clear();
            self.pending.sensors.clear();
            return;
        }
        self.update_broad_phase();
        self.update_contacts();
        self.solve(dt);
        // The broad-phase boxes follow the bodies at once, so that continuous collision below and
        // every query between steps (rays, casts, overlaps, the character's sweeps) see each
        // collider where it is now, not where the step started. Colliders that left their box
        // are queued for pair finding at the next step.
        let moved = std::mem::take(&mut self.scratch.members);
        for &b in &moved {
            self.refresh_body_proxies(b, false);
        }
        if self.settings.enable_continuous && !moved.is_empty() {
            self.continuous(&moved, dt);
        }
        self.scratch.members = moved;
        std::mem::swap(&mut self.events, &mut self.pending);
        self.pending.contacts.clear();
        self.pending.sensors.clear();
    }

    /// Finds new pairs for the colliders that moved out of their boxes (or were added or
    /// teleported). Boxes themselves are refreshed where bodies move: at the end of each step and
    /// in [`Self::set_body_pose`].
    pub(super) fn update_broad_phase(&mut self) {
        if self.move_buffer.is_empty() {
            return;
        }
        let mut moved = std::mem::take(&mut self.move_buffer);
        moved.sort_unstable();
        moved.dedup();
        for &c in &moved {
            self.colliders[c as usize].queued = false;
        }
        let mut found = std::mem::take(&mut self.scratch.broad_found);
        found.clear();
        for &c in &moved {
            let col = &self.colliders[c as usize];
            if !col.alive {
                continue;
            }
            let fat = col.fat;
            let mut visit = |o: u32| {
                if o != c {
                    let key = (c.min(o), c.max(o));
                    if !self.pairs.contains_key(&key) && self.should_collide(key.0, key.1) {
                        found.push(key);
                    }
                }
            };
            self.moving_tree.query(&fat, &mut |_, o| {
                visit(o);
                true
            });
            if !col.in_static_tree {
                self.static_tree.query(&fat, &mut |_, o| {
                    visit(o);
                    true
                });
            }
        }
        found.sort_unstable();
        found.dedup();
        for &(a, b) in &found {
            let sensor = self.colliders[a as usize].sensor || self.colliders[b as usize].sensor;
            let k = self.contacts.len() as u32;
            let slots = [a, b].map(|c| {
                let list = &mut self.colliders[c as usize].contacts;
                list.push(k);
                (list.len() - 1) as u32
            });
            self.contacts.push(Contact {
                a,
                b,
                manifolds: Vec::new(),
                touching: false,
                sensor,
                slots,
            });
            self.pairs.insert((a, b), k);
        }
        found.clear();
        self.scratch.broad_found = found;
        moved.clear();
        self.move_buffer = moved;
    }

    /// Updates a body's collider boxes; queues those that left their fat box (or all, when
    /// `force`) for pair finding.
    pub(super) fn refresh_body_proxies(&mut self, b: u32, force: bool) {
        let margin = self.aabb_margin();
        let len = self.bodies[b as usize].colliders.len();
        for ci in 0..len {
            let c = self.bodies[b as usize].colliders[ci];
            let tight = self.collider_box(c);
            let col = &self.colliders[c as usize];
            if force || !col.fat.contains(&tight) {
                let fat = tight.expand(margin);
                let (proxy, static_tree) = (col.proxy, col.in_static_tree);
                if static_tree {
                    self.static_tree.update(proxy, fat);
                } else {
                    self.moving_tree.update(proxy, fat);
                }
                self.colliders[c as usize].fat = fat;
                self.queue(c);
            }
        }
    }

    pub(super) fn should_collide(&self, a: u32, b: u32) -> bool {
        let (ca, cb) = (&self.colliders[a as usize], &self.colliders[b as usize]);
        if !ca.alive || !cb.alive || ca.body == cb.body {
            return false;
        }
        if ca.layers & cb.mask == 0 || cb.layers & ca.mask == 0 {
            return false;
        }
        let (ba, bb) = (
            &self.bodies[ca.body as usize],
            &self.bodies[cb.body as usize],
        );
        if ca.sensor || cb.sensor {
            // Sensors see moving bodies, never static ones or other sensors.
            if ca.sensor && cb.sensor {
                return false;
            }
            let other = if ca.sensor { bb } else { ba };
            return other.moves();
        }
        if !ba.is_dynamic() && !bb.is_dynamic() {
            return false;
        }
        if !ca.shape.is_convex() && !cb.shape.is_convex() {
            return false;
        }
        // A joint between the two bodies may turn their collisions off: it is in both bodies'
        // lists, so the shorter one is searched (a level body with many joints touches many
        // bodies with none).
        let joints = if ba.joints.len() <= bb.joints.len() {
            &ba.joints
        } else {
            &bb.joints
        };
        for &j in joints {
            let d = &self.joints[j as usize].desc;
            let (x, y) = (d.body_a.index, d.body_b.index);
            if !d.collide_connected
                && ((x == ca.body && y == cb.body) || (x == cb.body && y == ca.body))
            {
                return false;
            }
        }
        true
    }

    pub(super) fn destroy_contact(&mut self, k: usize, report: bool) {
        let c = &self.contacts[k];
        let key = (c.a, c.b);
        if report && c.touching {
            let (a, b) = (self.collider_id(c.a), self.collider_id(c.b));
            if c.sensor {
                let (sensor, visitor) = if self.colliders[c.a as usize].sensor {
                    (a, b)
                } else {
                    (b, a)
                };
                self.pending
                    .sensors
                    .push(SensorEvent::End { sensor, visitor });
            } else {
                self.pending.contacts.push(ContactEvent::End { a, b });
            }
        }
        // Out of its colliders' lists: the last entry of each list takes its place there.
        let slots = self.contacts[k].slots;
        for (end, c) in [key.0, key.1].into_iter().enumerate() {
            let s = slots[end] as usize;
            let list = &mut self.colliders[c as usize].contacts;
            list.swap_remove(s);
            if let Some(&m) = list.get(s) {
                let mc = &mut self.contacts[m as usize];
                mc.slots[usize::from(mc.a != c)] = s as u32;
            }
        }
        self.pairs.remove(&key);
        self.contacts.swap_remove(k);
        // The last contact now has index `k`: in the pair map and in its colliders' lists.
        if let Some(m) = self.contacts.get(k) {
            let (moved, slots) = ((m.a, m.b), m.slots);
            self.pairs.insert(moved, k as u32);
            self.colliders[moved.0 as usize].contacts[slots[0] as usize] = k as u32;
            self.colliders[moved.1 as usize].contacts[slots[1] as usize] = k as u32;
        }
    }

    /// The narrow phase: new manifolds for every pair with an awake moving body, carried
    /// impulses, begin and end events; pairs whose boxes no longer overlap are dropped.
    pub(super) fn update_contacts(&mut self) {
        let awake_count = self.awake_body_count();
        if awake_count == 0 {
            return;
        }
        let mut narrow_manifolds = std::mem::take(&mut self.scratch.narrow_manifolds);
        let mut old_manifolds = std::mem::take(&mut self.scratch.old_manifolds);
        let mut wake = std::mem::take(&mut self.scratch.wake);
        let mut active_indices = std::mem::take(&mut self.scratch.active_indices);
        wake.clear();

        if awake_count.saturating_mul(8) < self.contacts.len() {
            active_indices.clear();
            for (wi, &word) in self.awake_words.iter().enumerate() {
                let mut w = word;
                let base = (wi << 6) as u32;
                while w != 0 {
                    let b = (base + w.trailing_zeros()) as usize;
                    for &c in &self.bodies[b].colliders {
                        active_indices.extend_from_slice(&self.colliders[c as usize].contacts);
                    }
                    w &= w - 1;
                }
            }
            active_indices.sort_unstable();
            active_indices.dedup();

            let mut idx = 0;
            while idx < active_indices.len() {
                let k = active_indices[idx] as usize;
                if !self.update_single_contact(
                    k,
                    &mut narrow_manifolds,
                    &mut old_manifolds,
                    &mut wake,
                ) {
                    let last = self.contacts.len();
                    if k < last && active_indices.last() == Some(&(last as u32)) {
                        active_indices.pop();
                    } else {
                        idx += 1;
                    }
                    continue;
                }
                idx += 1;
            }
        } else {
            let mut k = 0;
            while k < self.contacts.len() {
                let (a, b) = (self.contacts[k].a, self.contacts[k].b);
                let (ca, cb) = (&self.colliders[a as usize], &self.colliders[b as usize]);
                let (ba, bb) = (
                    &self.bodies[ca.body as usize],
                    &self.bodies[cb.body as usize],
                );
                let active = (ba.awake && ba.moves()) || (bb.awake && bb.moves());
                if !active {
                    k += 1;
                    continue;
                }
                if !self.update_single_contact(
                    k,
                    &mut narrow_manifolds,
                    &mut old_manifolds,
                    &mut wake,
                ) {
                    continue;
                }
                k += 1;
            }
        }

        // A moving kinematic body wakes what hangs from it by a joint (islands leave kinematic
        // bodies out, and so does waking along joints).
        if self.kinematic_count > 0 {
            for j in &self.joints {
                if !j.alive {
                    continue;
                }
                let (x, y) = (j.desc.body_a.index, j.desc.body_b.index);
                for (p, q) in [(x, y), (y, x)] {
                    let (bp, bq) = (&self.bodies[p as usize], &self.bodies[q as usize]);
                    if bp.awake && bp.is_kinematic_moving() && !bq.awake && bq.is_dynamic() {
                        wake.push(q);
                    }
                }
            }
        }
        self.wake_islands(&wake);

        self.scratch.narrow_manifolds = narrow_manifolds;
        self.scratch.old_manifolds = old_manifolds;
        self.scratch.wake = wake;
        self.scratch.active_indices = active_indices;
    }

    /// Updates active contact `k`; returns `false` if the contact was destroyed.
    pub(super) fn update_single_contact(
        &mut self,
        k: usize,
        narrow_manifolds: &mut Vec<Manifold>,
        old_manifolds: &mut Vec<ContactManifold>,
        wake: &mut Vec<u32>,
    ) -> bool {
        let (a, b) = (self.contacts[k].a, self.contacts[k].b);
        let (ca, cb) = (&self.colliders[a as usize], &self.colliders[b as usize]);
        if !ca.fat.overlaps(&cb.fat) || !self.should_collide(a, b) {
            self.destroy_contact(k, true);
            return false;
        }
        let (iso_a, iso_b) = (self.collider_iso(a), self.collider_iso(b));
        narrow_manifolds.clear();
        let sensor = self.contacts[k].sensor;
        let speculative = if sensor {
            0.0
        } else {
            self.settings.speculative_distance
        };
        collide::collide(
            &self.colliders[a as usize].shape,
            &iso_a,
            &self.colliders[b as usize].shape,
            &iso_b,
            speculative,
            narrow_manifolds,
        );
        let was_touching = self.contacts[k].touching;
        let touching = if sensor {
            narrow_manifolds.iter().any(|m| m.min_separation() < 0.0)
        } else {
            narrow_manifolds.iter().any(|m| m.count > 0)
        };
        if !sensor {
            let mut manifolds = std::mem::take(&mut self.contacts[k].manifolds);
            std::mem::swap(&mut manifolds, old_manifolds);
            let match_distance =
                2.0 * self.settings.speculative_distance + self.settings.linear_slop;
            carry_impulses_into(
                old_manifolds,
                narrow_manifolds,
                &iso_a,
                match_distance,
                &mut manifolds,
            );
            self.contacts[k].manifolds = manifolds;
        } else {
            self.contacts[k].manifolds.clear();
        }
        self.contacts[k].touching = touching;
        if touching != was_touching {
            let (ia, ib) = (self.collider_id(a), self.collider_id(b));
            if sensor {
                let (sensor_id, visitor) = if self.colliders[a as usize].sensor {
                    (ia, ib)
                } else {
                    (ib, ia)
                };
                self.pending.sensors.push(if touching {
                    SensorEvent::Begin {
                        sensor: sensor_id,
                        visitor,
                    }
                } else {
                    SensorEvent::End {
                        sensor: sensor_id,
                        visitor,
                    }
                });
            } else {
                self.pending.contacts.push(if touching {
                    ContactEvent::Begin { a: ia, b: ib }
                } else {
                    ContactEvent::End { a: ia, b: ib }
                });
            }
        }
        if touching && !sensor {
            // An awake body that can push wakes a sleeping one it touches: a dynamic body, or
            // a kinematic one that moves. A kinematic body standing still does not (it would
            // wake its load the step after the load fell asleep, every time), unless the
            // contact has just begun (it was put there, or was given a new collider there).
            let (pa, pb) = (
                self.colliders[a as usize].body,
                self.colliders[b as usize].body,
            );
            for (p, q) in [(pa, pb), (pb, pa)] {
                let (bp, bq) = (&self.bodies[p as usize], &self.bodies[q as usize]);
                if bp.awake && (bp.pushes() || !was_touching) && !bq.awake && bq.is_dynamic() {
                    wake.push(q);
                }
            }
        }
        true
    }

    /// The soft step over every awake moving body; stores those bodies in `self.scratch.members`
    /// in ascending index order.
    pub(super) fn solve(&mut self, dt: f32) {
        self.rescale_warm_starts(self.settings.substeps.max(1));
        let mut scratch = std::mem::take(&mut self.scratch);
        self.collect_awake_moving(&mut scratch.members);
        if scratch.members.is_empty() {
            self.scratch = scratch;
            return;
        }

        let s = &self.settings;
        let substeps = s.substeps.max(1);
        let h = dt / substeps as f32;
        let contact_hertz = s.contact_hertz.min(0.25 / h);
        let joint_hertz = s.joint_hertz.min(0.25 / h);
        let ctx = StepContext {
            dt,
            h,
            inv_h: 1.0 / h,
            contact_softness: Softness::new(contact_hertz, s.contact_damping_ratio, h),
            static_softness: Softness::new(2.0 * contact_hertz, s.contact_damping_ratio, h),
            joint_softness: Softness::new(joint_hertz, s.joint_damping_ratio, h),
            max_push_velocity: s.contact_push_max_velocity,
            restitution_threshold: s.restitution_threshold,
        };

        // Solver bodies: slot 0 is static; then every awake moving body in index order.
        if scratch.slot.len() < self.bodies.len() {
            scratch.slot.resize(self.bodies.len(), 0);
        }
        scratch.solver_bodies.clear();
        scratch.solver_bodies.push(SolverBody::STATIC);
        scratch.solver_vels.clear();
        scratch.solver_vels.push(SolverVel::ZERO);
        scratch.per_body.clear();
        scratch.turning.clear();
        scratch.turning.push(Turning::Fixed);

        for &bi in &scratch.members {
            let i = bi as usize;
            let b = &mut self.bodies[i];
            b.center0 = b.center;
            b.rotation0 = b.rotation;
            scratch.slot[i] = scratch.solver_bodies.len();
            // Every setter keeps locked axes at zero already; this makes sure of it, since
            // nothing in the solve could take such a velocity away again.
            let (v, w) = b.allowed_velocities(b.linear_velocity, b.angular_velocity);
            let inv_inertia = b.world_inv_inertia();
            scratch.solver_vels.push(SolverVel { v, w });
            scratch.solver_bodies.push(SolverBody {
                dp: Vec3::ZERO,
                dq: Quat::IDENTITY,
                inv_mass: b.inv_mass_vector(),
                inv_inertia,
                contact_inv_inertia: inv_inertia,
            });
            scratch.turning.push(b.turning());
            scratch.per_body.push(BodyInput {
                force: b.force,
                torque: b.torque,
                gravity_scale: b.gravity_scale,
                linear_damping: 1.0 / (1.0 + h * b.linear_damping),
                angular_damping: 1.0 / (1.0 + h * b.angular_damping),
                kinematic_turn: (b.kind == BodyKind::Kinematic)
                    .then(|| Quat::from_angular_velocity(w, h)),
            });
        }

        // Contact constraints.
        scratch.constraints.clear();
        let build_contact_constraint =
            |w: &PhysicsWorld,
             k: usize,
             slot: &[usize],
             solver_bodies: &[SolverBody],
             solver_vels: &[SolverVel],
             constraints: &mut Vec<ContactConstraint>| {
                let c = &w.contacts[k];
                if !c.touching || c.sensor {
                    return;
                }
                let (ca, cb) = (&w.colliders[c.a as usize], &w.colliders[c.b as usize]);
                let ia = slot[ca.body as usize];
                let ib = slot[cb.body as usize];
                if ia == 0 && ib == 0 {
                    return;
                }
                let ba = &w.bodies[ca.body as usize];
                let bb = &w.bodies[cb.body as usize];
                let fa = BodyFrame {
                    center: ba.center,
                    rotation: ba.rotation,
                    local_center: ba.local_center,
                    index: ia,
                };
                let fb = BodyFrame {
                    center: bb.center,
                    rotation: bb.rotation,
                    local_center: bb.local_center,
                    index: ib,
                };
                let friction = (ca.friction * cb.friction).sqrt();
                let restitution = ca.restitution.max(cb.restitution);
                let either_static = solver_bodies[ia].inv_mass == Vec3::ZERO
                    || solver_bodies[ib].inv_mass == Vec3::ZERO;
                for (mi, m) in c.manifolds.iter().enumerate() {
                    if m.manifold.count == 0 {
                        continue;
                    }
                    let n = m.manifold.normal;
                    let (t1, t2) = n.orthonormal_basis();
                    let mut cc = ContactConstraint {
                        a: ia,
                        b: ib,
                        normal: n,
                        tangents: [t1, t2],
                        friction,
                        restitution,
                        softness: if either_static {
                            ctx.static_softness
                        } else {
                            ctx.contact_softness
                        },
                        points: [ConstraintPoint::default(); 4],
                        count: usize::from(m.manifold.count),
                        source: (k as u32, mi as u32),
                        rounding: (ca.shape.radius(), cb.shape.radius()),
                    };
                    for i in 0..usize::from(m.manifold.count) {
                        let p = m.manifold.points[i];
                        let cache = m.cache[i];
                        cc.points[i] = solver::prepare_point(
                            &cc,
                            solver_bodies,
                            solver_vels,
                            &fa,
                            &fb,
                            p.point,
                            p.separation,
                            cache.normal_impulse,
                            cache.friction_impulse,
                        );
                    }
                    constraints.push(cc);
                }
            };

        if scratch.members.len().saturating_mul(8) < self.contacts.len() {
            scratch.active_indices.clear();
            for &b in &scratch.members {
                for &c in &self.bodies[b as usize].colliders {
                    scratch
                        .active_indices
                        .extend_from_slice(&self.colliders[c as usize].contacts);
                }
            }
            scratch.active_indices.sort_unstable();
            scratch.active_indices.dedup();
            for &k in &scratch.active_indices {
                build_contact_constraint(
                    self,
                    k as usize,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &scratch.solver_vels,
                    &mut scratch.constraints,
                );
            }
        } else {
            for k in 0..self.contacts.len() {
                build_contact_constraint(
                    self,
                    k,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &scratch.solver_vels,
                    &mut scratch.constraints,
                );
            }
        }

        // Joint constraints.
        scratch.joint_constraints.clear();
        let build_joint_constraint =
            |w: &PhysicsWorld,
             j: usize,
             slot: &[usize],
             solver_bodies: &[SolverBody],
             joint_constraints: &mut Vec<(u32, JointConstraint)>| {
                let slot_j = &w.joints[j];
                if !slot_j.alive {
                    return;
                }
                let (Some(a), Some(b)) = (
                    w.body_index(slot_j.desc.body_a),
                    w.body_index(slot_j.desc.body_b),
                ) else {
                    return;
                };
                let ia = slot[a];
                let ib = slot[b];
                if ia == 0 && ib == 0 {
                    return;
                }
                let ba = &w.bodies[a];
                let bb = &w.bodies[b];
                let fa = BodyFrame {
                    center: ba.center,
                    rotation: ba.rotation,
                    local_center: ba.local_center,
                    index: ia,
                };
                let fb = BodyFrame {
                    center: bb.center,
                    rotation: bb.rotation,
                    local_center: bb.local_center,
                    index: ib,
                };
                let jc =
                    joint::prepare(&slot_j.desc, &fa, &fb, solver_bodies, &ctx, slot_j.impulses);
                joint_constraints.push((j as u32, jc));
            };

        if scratch.members.len().saturating_mul(8) < self.joints.len() {
            scratch.active_indices.clear();
            for &b in &scratch.members {
                scratch
                    .active_indices
                    .extend_from_slice(&self.bodies[b as usize].joints);
            }
            scratch.active_indices.sort_unstable();
            scratch.active_indices.dedup();
            for &j in &scratch.active_indices {
                build_joint_constraint(
                    self,
                    j as usize,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &mut scratch.joint_constraints,
                );
            }
        } else {
            for j in 0..self.joints.len() {
                build_joint_constraint(
                    self,
                    j,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &mut scratch.joint_constraints,
                );
            }
        }

        let inputs = IntegrationInputs {
            gravity: self.settings.gravity,
            per_body: &scratch.per_body,
            max_linear_speed: self.settings.max_linear_speed,
            max_angular_speed: 0.25 * std::f32::consts::PI * ctx.inv_h,
        };
        for sub in 0..substeps {
            solver::integrate_velocities(
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
                &inputs,
                h,
            );
            for (_, jc) in &scratch.joint_constraints {
                joint::warm_start(jc, &scratch.solver_bodies, &mut scratch.solver_vels);
            }
            solver::warm_start_contacts(
                &scratch.constraints,
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
            );
            for (j, jc) in scratch.joint_constraints.iter_mut() {
                let desc = self.joints[*j as usize].desc;
                joint::solve(
                    jc,
                    &desc,
                    &scratch.solver_bodies,
                    &mut scratch.solver_vels,
                    &ctx,
                    true,
                );
            }
            solver::solve_contacts(
                &mut scratch.constraints,
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
                &ctx,
                true,
                sub % 2 == 1,
            );
            solver::integrate_positions(
                &mut scratch.solver_bodies,
                &scratch.solver_vels,
                &inputs,
                h,
            );
            for (sb, t) in scratch.solver_bodies.iter_mut().zip(&scratch.turning) {
                if let Some(m) = t.after(sb.dq) {
                    sb.inv_inertia = m;
                }
            }
            for (j, jc) in scratch.joint_constraints.iter_mut() {
                let desc = self.joints[*j as usize].desc;
                joint::solve(
                    jc,
                    &desc,
                    &scratch.solver_bodies,
                    &mut scratch.solver_vels,
                    &ctx,
                    false,
                );
            }
            solver::solve_contacts(
                &mut scratch.constraints,
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
                &ctx,
                false,
                sub % 2 == 0,
            );
        }
        solver::apply_restitution(
            &mut scratch.constraints,
            &scratch.solver_bodies,
            &mut scratch.solver_vels,
            &ctx,
        );

        // Store impulses for warm starting, and report hits.
        let hit_threshold = self.settings.hit_event_threshold;
        scratch.hits.clear();
        for cc in &scratch.constraints {
            let (k, mi) = (cc.source.0 as usize, cc.source.1 as usize);
            let cm = &mut self.contacts[k].manifolds[mi];
            for i in 0..cc.count {
                let p = &cc.points[i];
                cm.cache[i].normal_impulse = p.normal_impulse;
                cm.cache[i].friction_impulse =
                    cc.tangents[0] * p.tangent_impulse[0] + cc.tangents[1] * p.tangent_impulse[1];
                let approach = -p.relative_velocity;
                if approach > hit_threshold && p.max_normal_impulse > 0.0 {
                    let point = cm.manifold.points[i].point;
                    let ku = k as u32;
                    if let Some(last) = scratch.hits.last_mut().filter(|last| last.0 == ku) {
                        if approach > last.1 {
                            *last = (ku, approach, point, cc.normal);
                        }
                    } else {
                        scratch.hits.push((ku, approach, point, cc.normal));
                    }
                }
            }
        }
        for &(k, speed, point, normal) in &scratch.hits {
            let c = &self.contacts[k as usize];
            self.pending.contacts.push(ContactEvent::Hit {
                a: self.collider_id(c.a),
                b: self.collider_id(c.b),
                point,
                normal,
                speed,
            });
        }
        for (j, jc) in &scratch.joint_constraints {
            self.joints[*j as usize].impulses = jc.impulses;
        }

        // Write the bodies back and clear their temporary slot assignments.
        for &b in &scratch.members {
            let sidx = scratch.slot[b as usize];
            scratch.slot[b as usize] = 0;
            let sb = &scratch.solver_bodies[sidx];
            let sv = scratch.solver_vels[sidx];
            let body = &mut self.bodies[b as usize];
            body.linear_velocity = sv.v;
            body.angular_velocity = sv.w;
            body.center = body.center0 + sb.dp;
            body.rotation = body.reachable((sb.dq * body.rotation0).normalize());
            body.position = body.center - body.rotation.rotate(body.local_center);
            body.force = Vec3::ZERO;
            body.torque = Vec3::ZERO;
        }

        self.scratch = scratch;
        if self.settings.enable_sleep {
            self.update_sleep(dt);
        }
    }

    /// The impulses carried over from the last step warm-start each substep, so they are per
    /// substep: when the substep count changes, they are rescaled to the new one.
    pub(super) fn rescale_warm_starts(&mut self, substeps: u32) {
        let before = std::mem::replace(&mut self.warm_substeps, substeps);
        if before == 0 || before == substeps {
            return;
        }
        let scale = before as f32 / substeps as f32;
        for c in &mut self.contacts {
            for m in &mut c.manifolds {
                for p in &mut m.cache {
                    p.normal_impulse *= scale;
                    p.friction_impulse *= scale;
                }
            }
        }
        for j in &mut self.joints {
            let i = &mut j.impulses;
            i.linear *= scale;
            i.angular *= scale;
            i.axial *= scale;
            i.lower *= scale;
            i.upper *= scale;
            i.motor *= scale;
        }
    }

    /// Sleep timers, then islands (bodies linked by touching contacts and joints) that have all
    /// been still for `time_to_sleep` go to sleep together. Runs in `O(awake)` using adjacency
    /// traversal instead of scanning all sleeping bodies and contacts in the world.
    pub(super) fn update_sleep(&mut self, dt: f32) {
        let mut scratch = std::mem::take(&mut self.scratch);
        let threshold = self.settings.sleep_threshold;
        for &b in &scratch.members {
            let body = &mut self.bodies[b as usize];
            let still = body.linear_velocity.length() < threshold
                && body.angular_velocity.length() * body.max_extent < threshold;
            if body.can_sleep && still {
                body.sleep_time += dt;
            } else {
                body.sleep_time = 0.0;
            }
        }
        // A moving kinematic body keeps what it touches, and what hangs from it by a joint,
        // awake, however slowly it moves (islands leave kinematic bodies out).
        if self.kinematic_count > 0 {
            scratch.sleep_kept.clear();
            for (p, bp) in self.bodies.iter().enumerate() {
                if bp.alive && bp.is_kinematic_moving() {
                    self.push_neighbours(p as u32, &mut scratch.sleep_kept);
                }
            }
            for &q in &scratch.sleep_kept {
                self.bodies[q as usize].sleep_time = 0.0;
            }
        }

        scratch.sleep_stamp = scratch.sleep_stamp.wrapping_add(1);
        if scratch.sleep_stamp == 0 {
            scratch.sleep_visited.fill(0);
            scratch.sleep_stamp = 1;
        }
        if scratch.sleep_visited.len() < self.bodies.len() {
            scratch.sleep_visited.resize(self.bodies.len(), 0);
        }
        let stamp = scratch.sleep_stamp;
        let time_to_sleep = self.settings.time_to_sleep;

        for mi in 0..scratch.members.len() {
            let b = scratch.members[mi];
            let body = &self.bodies[b as usize];
            if !body.is_dynamic() {
                // A kinematic body sleeps on its own once it has stopped.
                if body.sleep_time > 0.0
                    && body.sleep_time >= time_to_sleep
                    && body.linear_velocity == Vec3::ZERO
                    && body.angular_velocity == Vec3::ZERO
                {
                    self.bodies[b as usize].awake = false;
                    self.set_awake_bit(b, false);
                }
                continue;
            }
            if scratch.sleep_visited[b as usize] == stamp {
                continue;
            }

            scratch.sleep_visited[b as usize] = stamp;
            scratch.sleep_stack.clear();
            scratch.sleep_island.clear();
            scratch.sleep_stack.push(b);
            let mut min_sleep = f32::INFINITY;

            while let Some(u) = scratch.sleep_stack.pop() {
                scratch.sleep_island.push(u);
                let bu = &self.bodies[u as usize];
                min_sleep = min_sleep.min(if bu.can_sleep { bu.sleep_time } else { 0.0 });

                for &c in &bu.colliders {
                    for &k in &self.colliders[c as usize].contacts {
                        let ct = &self.contacts[k as usize];
                        if !ct.touching || ct.sensor {
                            continue;
                        }
                        let v = self.colliders[ct.other(c) as usize].body;
                        let bv = &self.bodies[v as usize];
                        if !bv.alive || !bv.is_dynamic() {
                            continue;
                        }
                        if !bv.awake {
                            // Touching an already-asleep dynamic body prevents sleeping here;
                            // waking has priority.
                            min_sleep = 0.0;
                        } else if scratch.sleep_visited[v as usize] != stamp {
                            scratch.sleep_visited[v as usize] = stamp;
                            scratch.sleep_stack.push(v);
                        }
                    }
                }
                for &j in &bu.joints {
                    let d = &self.joints[j as usize].desc;
                    for v in [d.body_a.index, d.body_b.index] {
                        if v == u {
                            continue;
                        }
                        let bv = &self.bodies[v as usize];
                        if !bv.alive || !bv.is_dynamic() {
                            continue;
                        }
                        if !bv.awake {
                            min_sleep = 0.0;
                        } else if scratch.sleep_visited[v as usize] != stamp {
                            scratch.sleep_visited[v as usize] = stamp;
                            scratch.sleep_stack.push(v);
                        }
                    }
                }
            }

            // Still for at least one step, whatever time_to_sleep is (zero included).
            if min_sleep > 0.0 && min_sleep >= time_to_sleep {
                for &u in &scratch.sleep_island {
                    let body = &mut self.bodies[u as usize];
                    body.awake = false;
                    body.linear_velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                    self.set_awake_bit(u, false);
                }
            }
        }
        self.scratch = scratch;
    }
}
